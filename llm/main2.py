from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
model = None
tokenizer = None


@app.on_event("startup")
def startup_event():
    global model, tokenizer
    hf_token = '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'
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it", token=hf_token)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-27b-it", token=hf_token, device_map="auto")


class InputText(BaseModel):
    text: str


@app.post("/send/")
async def send_response(input: InputText):

    if torch.cuda.is_available():
        input_ids = tokenizer(input.text, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_length=4000)
        res = tokenizer.decode(outputs[0]).replace('<bos>', '').replace('<eos>', '')
    else:
        res = input.text

    return {
        'message': res
    }