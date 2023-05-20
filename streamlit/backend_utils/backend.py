from fastapi import FastAPI, File
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoConfig, GPT2Tokenizer
from transformers import MBartForConditionalGeneration, MBartConfig, MBartTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from pydantic import BaseModel
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Item(BaseModel):
    text: str
    min_length: int
    max_length: int
    length_penalty: float
    no_repeat_ngram_size: int
    repetition_penalty: float
    top_k: int
    top_p: float
    num_beams: int
    temperature: float
    model_name: str

app = FastAPI()
@app.post("/summarize/")
async def summarize(item: Item):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if item.model_name == "ruT5-base":
        config = T5Config.from_pretrained('IlyaGusev/rut5_base_sum_gazeta')
        
        model = T5ForConditionalGeneration(config=config)
        model.load_state_dict(torch.load('/Users/pavelkulpin/Downloads/model_states/rut5_base_sum_gazeta_state.pth', map_location=device))
        # model.to(device)
        tokenizer = T5Tokenizer.from_pretrained('IlyaGusev/rut5_base_sum_gazeta')
    
    elif item.model_name == "FRED-T5-Large":
        config = AutoConfig.from_pretrained('ai-forever/FRED-T5-large')
        
        model = T5ForConditionalGeneration(config=config)
        model.load_state_dict(torch.load('/Users/pavelkulpin/Downloads/model_states/fred_model_state.pth', map_location=device))
        # model.to(device)
        model.to(device)
        
        tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-large', eos_token='</s>')
    
    elif item.model_name == "mBART":
        config = MBartConfig.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta')
        
        model = MBartForConditionalGeneration(config=config)
        model.load_state_dict(torch.load('/Users/pavelkulpin/Downloads/model_states/mbart_model_state.pth', map_location=device))
        # model.to(device)
        tokenizer = MBartTokenizer.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta')
    
    else:
        raise NameError('uknown model name: {}'.format(item.model_name))
    
    inputs = tokenizer.encode(item.text, return_tensors="pt")
    outputs = model.generate(inputs, 
                             max_length=item.max_length, 
                             min_length=item.min_length, 
                             length_penalty=item.length_penalty, 
                             num_beams=item.num_beams, 
                             no_repeat_ngram_size=item.no_repeat_ngram_size, 
                             repetition_penalty=item.repetition_penalty, 
                             top_k=item.top_k, 
                             top_p=item.top_p, 
                             temperature=item.temperature)
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"summary": summary}
