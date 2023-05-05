import nltk
import numpy as np
import torch
from tqdm import tqdm
import wandb
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


def preprocess_function(row, tokenizer, max_input_length, max_target_length, document_field_name, summary_field_name):
    model_inputs = tokenizer(text=row[document_field_name], max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=row[summary_field_name], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class T5SumDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_input_length, max_target_length, document_field_name, summary_field_name, ds_type='train'):
        super().__init__()
        assert path[-4:] == '.csv', 'dataset file is not a csv file'
        self.path = path
        self.ds_type = ds_type
        df = pd.read_csv(self.path)
        self.data = []
        for ind in tqdm(df.index, total=len(df), desc='Creating dataset'):
            self.data += [preprocess_function(df.loc[ind], tokenizer, max_input_length, max_target_length, document_field_name, summary_field_name)]

        del df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def t5_collate_batch(pad_id, batch):
    input_ids = []
    labels = []
    for sample in batch:
        input_ids.append(torch.tensor(sample['input_ids'], dtype=torch.long))
        labels.append(torch.tensor(sample['labels'], dtype=torch.long))

    batch = {
        'input_ids': pad_sequence(input_ids, padding_value=pad_id, batch_first=True),
        'labels': pad_sequence(labels, padding_value=-100, batch_first=True)
    }
    batch['attention_mask'] = (batch['input_ids'] != pad_id).clone()

    return batch


def compute_metrics(eval_pred, tokenizer, rouge_metric, bleu_metric):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    #BLEU
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    bleu_result = bleu_metric.compute(predictions=[decoded_preds], references=[[decoded_labels]], )

    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in rouge_result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    try:
        result["bleu"] = bleu_result['score']
    except KeyError:
        result["bleu"] = bleu_result['bleu']

    return {k: round(v, 4) for k, v in result.items()}


class T5Summarization(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def summarize(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            attention_mask=batch['attention_mask']
        )
        return outputs.loss

    def train_one_epoch(self, dataloader, optimizer, scheduler, accum_steps, use_mp, device, need_wandb, epoch, num_epochs, saving_steps_fraction, saving_dir):
        self.model.train()

        scaler = torch.cuda.amp.GradScaler() if use_mp else None

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Trainig {epoch}/{num_epochs}'):
            for k, v in batch.items():
                batch[k] = v.to(device)

            if use_mp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = self.summarize(batch)
                loss_item = loss.item()
                scaler.scale(loss).backward()
                if i % accum_steps == accum_steps - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss = self.summarize(batch)
                loss_item = loss.item()
                loss.backward()
                if i % accum_steps == accum_steps - 1:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if i % (len(dataloader) * saving_steps_fraction) == (len(dataloader) * saving_steps_fraction) - 1:
                if saving_dir == '':
                    torch.save(self.model.state_dict(), f'model_state_ep{epoch+1}_step{i}.pth')
                else:
                    if saving_dir[-1] != '/':
                        saving_dir += '/'
                    torch.save(self.model.state_dict(), saving_dir + f'model_state_ep{epoch+1}_step{i}.pth')
            if need_wandb:
                wandb.log({
                    'Loss': loss_item,
                    'Learning_Rate': scheduler.get_last_lr()[0]
                })

            scheduler.step()
            
    def evaluate(self, tokenizer, dataloader, rouge_metric, bleu_metric, device, min_length, max_length, need_wandb=False):
        self.model.eval()
        rouges1, rouges2, rougesL, rougesLsum, gen_len, bleu_ = [], [], [], [], [], []

        for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.no_grad():
                output_sequences = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=max_length, min_length=min_length, do_sample=False)
            
            metrics = compute_metrics((output_sequences.cpu(), batch['labels'].cpu()), tokenizer, rouge_metric, bleu_metric)
            
            rouges1.append(metrics['rouge1'])
            rouges2.append(metrics['rouge2'])
            rougesL.append(metrics['rougeL'])
            rougesLsum.append(metrics['rougeLsum'])
            gen_len.append(metrics['gen_len'])
            bleu_.append(metrics['bleu'])

        result = {'rouge1': np.mean(rouges1), 'rouge2': np.mean(rouges2),
                    'rougeL': np.mean(rougesL), 'rougeLsum': np.mean(rougesLsum),
                    'bleu': np.mean(bleu_), 'gen_len': np.mean(gen_len)}
        if need_wandb:
            wandb.log(result)

        return result