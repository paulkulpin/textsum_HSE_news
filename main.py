from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import get_cosine_schedule_with_warmup
import argparse
from functools import partial
import nltk
import numpy as np
import wandb
import evaluate
import warnings
import json

from scripts.models.rut5_sum import ruT5SumDataset, ruT5Summarization, rut5_collate_batch
from scripts.models.fred_sum import FREDSumDataset, FREDSummarization, fred_collate_batch
from scripts.models.mbart_sum import MBARTSumDataset, MBARTSummarization, mbart_collate_batch
from scripts.models.rugpt_sum import ruGPTSumDataset, ruGPTSummarization, rugpt_collate_batch
from scripts.models.mt5_sum import mT5SumDataset, mT5Summarization, mt5_collate_batch


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scrapped json.')
    parser.add_argument('--action', type=str, help="training/evaluating", default="evaluating")
    parser.add_argument('--model_type', type=str, help="ruT5/ruGPT/mBART/mT5/FRED", default="ruT5")
    #base params
    parser.add_argument('--csv_dataset_path', type=str, help="Full path to dataset csv file.", default="processed_news.csv")
    parser.add_argument('--shuffle_dataset', type=bool, help="if dataset suffling needed True/False", default=False)
    parser.add_argument('--reduce_dataset', type=float, help="Percentage part to reduce", default=100.0)
    parser.add_argument('--model_state_path', type=str, help="Full path to model state_dict.", default="")
    parser.add_argument('--res_model_state_path', type=str, help="Full path to resulting model state_dict.", default="model_state.pth")
    parser.add_argument('--res_model_comments_path', type=str, help="Path to json file with comments", default="model_comments.json")
    parser.add_argument('--HF_model_name', type=str, help="Hugging Face model/tokenizer name", default="")
    parser.add_argument('--random_seed', type=int, help="Random seed for torch, numpy.", default=101)
    parser.add_argument('--cuda_idx', type=int, help="cuda:idx", default=0)
    # additional base params
    parser.add_argument('--num_workers', type=int, help="Number of workers for dataloader, etc.", default=0)
    parser.add_argument('--source_text_field_name', type=str, help="Name of field that includes article.", default="document")
    parser.add_argument('--annotation_field_name', type=str, help="Name of field that includes annotation", default="summary")
    #wandb params
    parser.add_argument('--need_wandb', type=bool, help="if wandb logging needed True/False", default=False)
    parser.add_argument('--wandb_key', type=str, help="key token for wandb", default="")
    parser.add_argument('--wandb_project_name', type=str, help="Run Name", default="myproject")
    parser.add_argument('--wandb_run_name', type=str, help="Run Name", default="noname run")
    #training params
    parser.add_argument('--accum_steps', type=int, help="Accumulation Steps", default=1)
    parser.add_argument('--use_mp', type=bool, help="use GradScaler (not recommended) True/False", default=False)
    parser.add_argument('--batch_size', type=int, help="Batch Size", default=8)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training", default=1)
    parser.add_argument('--lr', type=int, help="Learning rate for AdamW optimizer", default=0.0001)
    parser.add_argument('--weight_decay', type=int, help="weight_decay for AdamW optimizer", default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, help="num_warmup_steps for cosine scheduler", default=1000)
    parser.add_argument('--num_cycles', type=int, help="num_cycles for cosine scheduler", default=0.5)
    parser.add_argument('--use_clipping', type=bool, help="if clipping needed True/False", default=False)

    parser.add_argument('--max_input_length', type=int, help="Max length of input sequence for dataset.", default=1000)
    parser.add_argument('--max_target_length', type=int, help="Max length of target sequence for dataset.", default=100)
    parser.add_argument('--eval_max_length', type=int, help="Max length of target sequence for generating.", default=150)
    parser.add_argument('--eval_min_length', type=int, help="Min length of target sequence for generating.", default=10)
    parser.add_argument('--ignore_warnings', type=bool, help="Ignore warnings? (True/False)", default=False)
    parser.add_argument('--dir_for_in_between_savings', type=str, help="Name of the directory to save in-between model states.", default="")
    parser.add_argument('--saving_steps_fraction', type=float, help="Fraction of steps for model.save per epoch.", default=0.1)

    args = dict(vars(parser.parse_args()))

    set_random_seed(args['random_seed'])
    print(f'>>>fixed random seed: {args["random_seed"]}\n')

    if args['ignore_warnings']:
        warnings.filterwarnings("ignore")
        print('>>>warnings will be ignored\n')

    assert args['csv_dataset_path'][-4:] == '.csv', f'dataset file [{args["csv_dataset_path"]}] must be .csv file.'
    assert args['res_model_comments_path'][-5:] == '.json', f'comments file [{args["res_model_comments_path"]}] must be .json file.'

    nltk.download('punkt')
    nltk.download('stopwords')
    print('>>>downloaded nltk corpuses.\n')

    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")
    print('>>>downloaded metrics.\n')

    device = torch.device(f'cuda:{args["cuda_idx"]}' if torch.cuda.is_available() else 'cpu')
    print(f'>>>device == {device}.\n')

    if args['HF_model_name'] == '':
        if args['model_type'] == 'ruT5':
            args['HF_model_name'] = 'IlyaGusev/rut5_base_sum_gazeta'
        elif args['model_type'] == 'FRED':
            args['HF_model_name'] = 'ai-forever/FRED-T5-large'
        elif args['model_type'] == 'MBART':
            args['HF_model_name'] = 'IlyaGusev/mbart_ru_sum_gazeta'
        elif args['model_type'] == 'ruGPT':
            args['HF_model_name'] = 'IlyaGusev/rugpt3medium_sum_gazeta'
        else:
            raise NotImplementedError(f'{args["model_type"]}')
    
    if args['model_type'] == 'ruT5':
        if args['model_state_path'] != "":
            conf = T5Config.from_pretrained(args['HF_model_name'])
            model = T5ForConditionalGeneration(config=conf)
            model.load_state_dict(torch.load(args['model_state_path']))
            model.to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args['HF_model_name']).to(device)

        print('>>>downloaded model.\n')
        tokenizer = T5Tokenizer.from_pretrained(args['HF_model_name'])
        print('>>>downloaded tokenizer.\n')

        dataset = ruT5SumDataset(args['csv_dataset_path'], 
                               tokenizer, 
                               args['max_input_length'], 
                               args['max_target_length'], 
                               args['source_text_field_name'], 
                               args['annotation_field_name'], 
                               args['reduce_dataset'])
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=partial(rut5_collate_batch, tokenizer.pad_token_id), 
                                                 batch_size=args['batch_size'], 
                                                 shuffle=args['shuffle_dataset'], 
                                                 pin_memory=True, 
                                                 num_workers=args['num_workers'])
        sum_model = ruT5Summarization(model)

    elif args['model_type'] == 'mT5':
        if args['model_state_path'] != "":
            conf = T5Config.from_pretrained(args['HF_model_name'])
            model = T5ForConditionalGeneration(config=conf)
            model.load_state_dict(torch.load(args['model_state_path']))
            model.to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args['HF_model_name']).to(device)

        print('>>>downloaded model.\n')
        tokenizer = T5Tokenizer.from_pretrained(args['HF_model_name'])
        print('>>>downloaded tokenizer.\n')

        dataset = mT5SumDataset(args['csv_dataset_path'], 
                               tokenizer, 
                               args['max_input_length'], 
                               args['max_target_length'], 
                               args['source_text_field_name'], 
                               args['annotation_field_name'], 
                               args['reduce_dataset'])
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=partial(mt5_collate_batch, tokenizer.pad_token_id), 
                                                 batch_size=args['batch_size'], 
                                                 shuffle=args['shuffle_dataset'], 
                                                 pin_memory=True, 
                                                 num_workers=args['num_workers'])
        sum_model = mT5Summarization(model)

    elif args['model_type'] == 'FRED':
        if args['model_state_path'] != "":
            conf = T5Config.from_pretrained(args['HF_model_name'])
            model = T5ForConditionalGeneration(config=conf)
            model.load_state_dict(torch.load(args['model_state_path']))
            model.to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args['HF_model_name']).to(device)

        print('>>>downloaded model.\n')
        tokenizer = GPT2Tokenizer.from_pretrained(args['HF_model_name'], eos_token='</s>')
        print('>>>downloaded tokenizer.\n')

        dataset = FREDSumDataset(args['csv_dataset_path'], 
                                 tokenizer, 
                                 args['max_input_length'], 
                                 args['max_target_length'], 
                                 args['source_text_field_name'], 
                                 args['annotation_field_name'], 
                                 args['reduce_dataset'])
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=partial(fred_collate_batch, tokenizer.pad_token_id), 
                                                 batch_size=args['batch_size'], 
                                                 shuffle=args['shuffle_dataset'], 
                                                 pin_memory=True, 
                                                 num_workers=args['num_workers'])
        sum_model = FREDSummarization(model)

    elif args['model_type'] == 'MBART':
        if args['model_state_path'] != "":
            conf = MBartConfig.from_pretrained(args['HF_model_name'])
            model = MBartForConditionalGeneration(config=conf)
            model.load_state_dict(torch.load(args['model_state_path']))
            model.to(device)
        else:
            model = MBartForConditionalGeneration.from_pretrained(args['HF_model_name']).to(device)

        print('>>>downloaded model.\n')
        tokenizer = MBartTokenizer.from_pretrained(args['HF_model_name'])
        print('>>>downloaded tokenizer.\n')

        dataset = MBARTSumDataset(args['csv_dataset_path'], 
                                  tokenizer, 
                                  args['max_input_length'], 
                                  args['max_target_length'], 
                                  args['source_text_field_name'], 
                                  args['annotation_field_name'], 
                                  args['reduce_dataset'])
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=partial(mbart_collate_batch, tokenizer.pad_token_id), 
                                                 batch_size=args['batch_size'], 
                                                 shuffle=args['shuffle_dataset'], 
                                                 pin_memory=True, 
                                                 num_workers=args['num_workers'])
        sum_model = MBARTSummarization(model)

    elif args['model_type'] == 'ruGPT':
        if args['model_state_path'] != "":
            conf = GPT2Config.from_pretrained(args['HF_model_name'])
            model = GPT2LMHeadModel(config=conf)
            model.load_state_dict(torch.load(args['model_state_path']))
            model.to(device)
        else:
            model = GPT2LMHeadModel.from_pretrained(args['HF_model_name']).to(device)

        print('>>>downloaded model.\n')
        tokenizer = GPT2Tokenizer.from_pretrained(args['HF_model_name'])
        print('>>>downloaded tokenizer.\n')
        
        if args['action'] == 'training':
            dataset = ruGPTSumDataset(args['csv_dataset_path'], 
                                    tokenizer, 
                                    args['max_input_length'], 
                                    args['max_target_length'], 
                                    args['source_text_field_name'], 
                                    args['annotation_field_name'], 
                                    args['reduce_dataset'], 
                                    ds_type='train')
        else:
            dataset = ruGPTSumDataset(args['csv_dataset_path'], 
                                    tokenizer, 
                                    args['max_input_length'], 
                                    args['max_target_length'], 
                                    args['source_text_field_name'], 
                                    args['annotation_field_name'], 
                                    ds_type='test')
        
        args['eval_max_length'] += args['max_input_length']
        args['eval_min_length'] += args['max_input_length']
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=partial(rugpt_collate_batch, tokenizer.pad_token_id), 
                                                 batch_size=args['batch_size'], 
                                                 shuffle=args['shuffle_dataset'], 
                                                 pin_memory=True, 
                                                 num_workers=args['num_workers'])
        sum_model = ruGPTSummarization(model)
    else:
        raise NotImplementedError(f'unknown model_type {args["model_type"]}')

    if args['need_wandb']:
        wandb.login(key=args['wandb_key'])
        wandb.init(project=args['wandb_project_name'], config=args, name=args['wandb_run_name'])

    if args['action'] == 'training':
        with open(args['res_model_comments_path'], 'w') as f:
            args['model_tp'] = 'T5ForConditionalGeneration'
            args['tokenizer_tp'] = 'T5Tokenizer'
            json.dump(args, f, indent="")

        optimizer = torch.optim.AdamW(sum_model.parameters(), 
                                      lr=args['lr'], 
                                      weight_decay=args['weight_decay'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args['num_warmup_steps'], 
                                                    num_training_steps=args['num_epochs'] * len(dataloader), 
                                                    num_cycles=args['num_cycles'])

        for epoch in range(args['num_epochs']):
            sum_model.train_one_epoch(dataloader, 
                                      optimizer, 
                                      scheduler, 
                                      accum_steps=args['accum_steps'], 
                                      use_mp=args['use_mp'], 
                                      device=device, 
                                      need_wandb=args['need_wandb'], 
                                      epoch=epoch, 
                                      num_epochs=args['num_epochs'], 
                                      saving_steps_fraction=args['saving_steps_fraction'], 
                                      saving_dir=args['dir_for_in_between_savings'], 
                                      use_clipping=args['use_clipping'])

        torch.save(model.state_dict(), args['res_model_state_path'])

    elif args['action'] == 'evaluating':
        eval_res = sum_model.evaluate(tokenizer=tokenizer, 
                                      dataloader=dataloader, 
                                      bleu_metric=bleu_metric, 
                                      rouge_metric=rouge_metric, 
                                      device=device, 
                                      need_wandb=args['need_wandb'], 
                                      min_length=args['eval_min_length'], 
                                      max_length=args['eval_max_length'])
        print('\n\n################ EVALUATION RESULT ################')
        print(eval_res)
        print('###################################################\n\n')
        if args['need_wandb']:
            wandb.log(eval_res)
    else:
        raise NotImplementedError(f'Unknown action: {args["action"]}')

    if args['need_wandb']:
        wandb.finish()
