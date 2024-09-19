import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO
import PIL
from tqdm.auto import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def fetch_and_open_image(image_id):
    try:
        url = f'https://i.pinimg.com/564x/{image_id[:2]}/{image_id[2:4]}/{image_id[4:6]}/{image_id}.jpg'
        #print(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.pinterest.com/',
        }
        
        with requests.Session() as session:
            response = session.get(url, headers=headers, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                img = Image.open(requests.get(url, stream=True).raw)
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

def load_image(image_file, input_size=448, max_num=12):
    res=fetch_and_open_image(image_file)
    res.save('images/{}.jpeg'.format(image_file))
    image = Image.open('images/{}.jpeg'.format(image_file)).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

def generate_res(model,tokenizer,title,desc,im):
    results=[]
    for tit,desc,im in tqdm(zip(title,description,image)):
        # set the max number of tiles in `max_num`
        pixel_values = load_image(im, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        # pure-text conversation (纯文本对话)
        question1 = f"""
    You are an excellent content annotator. You will be provided with detailed text and information of a shoppable product on some eCommerce website.
    Please provide more details about the product in terms of google product category, brand, color, material and gender.
    The output should be in the following format and only this:

    Google Product Category: Your response
    Brand: Your response
    Color: Your response
    Material: Your response
    Gender: Your response
    

    """.strip()
        question2="Here is the textual description of the shoppable product: Title: {}. Description: {}".format(tit,desc)
        question= question1 + '\n' + question2
        # response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')

        # question = 'Can you tell me a story?'
        # response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')

        # single-image single-round conversation (单图单轮对话)
        #question = '<image>\nPlease describe the image shortly.'
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        #print(f'User: {question}\nAssistant: {response}')
        print(response.strip())

        # # single-image multi-round conversation (单图多轮对话)
        # question = '<image>\nPlease describe the image in detail.'
        # response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        results.append(response.strip())
        # break
        
    return results




if __name__=='__main__':
    
    path = 'OpenGVLab/InternVL2-8B'
    device_map = split_model('InternVL2-8B')
    model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    


    
    df=pd.read_csv('Subsampled_eval_dataset.csv')
    title=df['title']
    description=df['description']
    image=df['image_signature']

    
    output=generate_res(model,tokenizer,title,description,image)
    df['InternVL2']=output
    df.to_csv('Subsampled_eval_dataset_InternVL2.csv', index=False)



