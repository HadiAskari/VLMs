
from transformers import pipeline
from tqdm.auto import tqdm
import pickle as pkl
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO
import PIL
import base64
import os


def generate_res(model,title,description,imageid):
    
    results=[]
    for tit,desc,im in tqdm(zip(title,description,imageid)):
        
        image = Image.open('../InternVL2-8b/images/{}.jpeg'.format(im)).convert('RGB')
        question = "Here is the textual description of the shoppable product: Title: {}. Description: {}".format(tit,desc)
        msgs = [{'role': 'user', 'content': question}]

        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False, # if sampling=False, beam_search will be used by default
            temperature=0.001,
            max_new_tokens=100,
            system_prompt=f"""
    You are an excellent content annotator. You will be provided with detailed text and information of a shoppable product on some eCommerce website.
    Please provide more details about the product in terms of google product category, brand, color, material and gender.
    The output should be in the following format and only this:

    Google Product Category: Your response
    Brand: Your response
    Color: Your response
    Material: Your response
    Gender: Your response
    

    """ # pass system_prompt if needed
        )
        
    
        print(res)
        results.append(res)
        
    
    return results




if __name__=='__main__':
    
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    model.eval()
        
    
    df=pd.read_csv('Subsampled_eval_dataset.csv')
    title=df['title']
    description=df['description']
    image=df['image_signature']

    
    output=generate_res(model,title,description,image)
    df['MiniCPM']=output
    df.to_csv('Subsampled_eval_dataset_MiniCPM.csv', index=False)



    