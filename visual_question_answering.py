import torch
import transformers
import torchvision
import pandas as pd
from PIL import Image
from torch import cuda, float16
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

auth_token = "your_hf_access_token"
tokenizer_id = 'lmsys/vicuna-7b-v1.5'
model_id = 'THUDM/cogagent-vqa-hf'
device = torch.device('cuda')


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=float16
)


def get_tokenizer_model(tokenizer_id, model_id, quantization_config):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config
    ).eval()
    return tokenizer, model


tokenizer, model = get_tokenizer_model(tokenizer_id, model_id, bnb_config)

def process_image(img_path):
  image = Image.open(img_path).convert('RGB')
  input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
  inputs = {
      'input_ids': input_by_model['input_ids'].unsqueeze(0).to('cuda'),
      'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to('cuda'),
      'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to('cuda'),
      'images': [[input_by_model['images'][0].to('cuda').to(torch.float16)]],
  }
  if 'cross_images' in input_by_model and input_by_model['cross_images']:
    inputs['cross_images'] = [[input_by_model['cross_images'][0].to('cuda').to(torch.float16)]]

  return inputs

def generate_outputs(inputs, gen_kwargs={"max_length": 2048, "do_sample": False}):
  with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(outputs[0])

image_path = input("Enter Image Path:")
query = input("Ask your Question:")
inputs = process_image('/content/Screenshot 2024-02-28 114231.png')
result = generate_outputs(inputs, gen_kwargs = {"max_length": 2048,
                "do_sample": False})

print(result)