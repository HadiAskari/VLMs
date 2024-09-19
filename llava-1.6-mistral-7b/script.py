
from transformers import pipeline
from tqdm.auto import tqdm
import pickle as pkl
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO
import PIL
import base64

def fetch_and_open_image(image_id):
    try:
        url = f'https://i.pinimg.com/564x/{image_id[:2]}/{image_id[2:4]}/{image_id[4:6]}/{image_id}.jpg'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.pinterest.com/',
        }
        
        with requests.Session() as session:
            response = Image.open(session.get(url, headers=headers, stream=True).raw)
            return response
            # Check if the request was successful
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # print('here')
                # print(img)
                return img
            elif response.status_code == 403:
                print("Access forbidden: You do not have permission to access this resource.")
            else:
                print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except PIL.UnidentifiedImageError:
        print("Cannot identify image file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return "Error"


def generate_res(model,title,description,image):
    
    results=[]
    for tit,desc,im in tqdm(zip(title,description,image)):
        conversation = [
        {"role": "user", "content": f"""
    You are an excellent content annotator. You will be provided with detailed text and information of a shoppable product on some eCommerce website.
    Please provide more details about the product in terms of google product category, brand, color, material and gender.
    The output should be in the following format and only this:

    Google Product Category: Your response
    Brand: Your response
    Color: Your response
    Material: Your response
    Gender: Your response
    

    """.strip()
    },
        {"role": "user", "content": "Here is the textual description of the shoppable product: Title: {}. Description: {}".format(tit,desc)},
    ]
        
        imm=fetch_and_open_image(im)
        #print(imm)
        # buffered = BytesIO()
        # imm.save(buffered, format="JPEG")
        # encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')   
        
        #print(encoded_image)     
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(prompt, imm, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=256)

        #print(processor.decode(output[0], skip_special_tokens=True))
        results.append(processor.decode(output[0], skip_special_tokens=True))
        
    
    return results




if __name__=='__main__':
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    
    print('what')

    
    df=pd.read_csv('Subsampled_eval_dataset.csv')
    title=df['title']
    description=df['description']
    image=df['image_signature']

    
    output=generate_res(model,title,description,image)
    df['Llava']=output
    df.to_csv('Subsampled_eval_dataset_Llava.csv', index=False)



    