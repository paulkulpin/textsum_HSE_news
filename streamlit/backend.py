from fastapi import FastAPI, File
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

app = FastAPI()
model_name = 'cointegrated/rut5-base-absum'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@app.post("/summarize/")
async def summarize(text: str, min_length: int = 40, max_length: int = 150):
    model_inputs = tokenizer(text=text, max_length=1000, truncation=True)
    model_inputs['input_ids'] = pad_sequence(torch.tensor(model_inputs['input_ids']).unsqueeze(0),
                                             padding_value=tokenizer.pad_token_id, batch_first=True)
    tok_sum = model.generate(input_ids=model_inputs['input_ids'].to(device),
                             max_length=max_length, min_length=min_length)
    summary = tokenizer.decode(tok_sum.squeeze(0), skip_special_tokens=True)
    return {"summary": summary}
