from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict

from PIL import Image
import requests
import copy
import torch
import shutil
import os
import time

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = True  # In this demo, we consider using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it.



warnings.filterwarnings("ignore")
pretrained = "/group/40034/auroraji/pretrained/LLaDA-V"
# pretrained = "/group/40034/auroraji/LLaDA-V/train/exp/llada_v_finetune_6"

model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()


if use_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")


data_list = [
    {
    "id": "000000312621",
    "image": "coco/train2017/000000312621.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat would you consider to be the most notable feature of the building's exterior, and why?"
      },
      {
        "from": "gpt",
        "value": "The most notable feature of the building's exterior is the large, red and white clock mounted on the side of the building above a high arch. The clock's prominent size and striking color make it an eye-catching focal point. Additionally, it is unique due to its location above a door and the presence of statues surrounding the clock. This design contributes to the building's aesthetic appeal and helps it stand out, making it easily recognizable and memorable."
      }
    ]
  },
  {
    "id": "000000565044",
    "image": "coco/train2017/000000565044.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What is a possible outcome of the woman's actions in this image?\n<image>"
      },
      {
        "from": "gpt",
        "value": "A potential outcome of the woman's actions in this image is that she successfully hits the tennis ball with her racket and sends it flying towards her opponent's side of the court. This would result in the tennis match continuing as the opponent attempts to return the ball. Her proper stance and grip on the racket suggest that she has good control and technique, increasing the likelihood of a well-executed shot."
      }
    ]
  },
  {
    "id": "000000372554",
    "image": "coco/train2017/000000372554.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat factors could affect the man's ability to successfully fly the octopus kite?"
      },
      {
        "from": "gpt",
        "value": "The man's ability to successfully fly the large orange octopus kite could be affected by various factors such as wind conditions, skill level, and the size of the kite relative to the open space available. In the image, the man is standing in a grassy area surrounded by other individuals, which suggests the possibility of limited space for maneuvering the kite. A large open space with minimal obstructions is usually more suitable for flying kites. Additionally, the wind conditions play a crucial role in determining the success of flying a kite - if the wind is too weak or too strong, it may be difficult to control the kite's movement and maintain stability in the air. Gaining expertise in managing kite flying also plays a role, as the individual's skill and experience will impact their ability to operate the kite effectively and make the necessary adjustments based on environmental factors."
      }
    ]
  },
  {
    "id": "000000261414",
    "image": "coco/train2017/000000261414.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What makes the image noteworthy?\n<image>"
      },
      {
        "from": "gpt",
        "value": "The image is noteworthy due to the unusual and amusing combination of elements involving the man surfing in the ocean. Instead of a typical surfing scene, the man is dressed in a wacky costume, decorated with souvenirs such as the Union Jack, and holding fishing rods. His appearance, along with him riding a surfboard with an Afro, makes the image stand out and captures the viewer's attention for its humor and unconventional aspects. It also highlights the creative and playful elements that can be found in edited images, pushing beyond the boundaries of ordinary surfing photos."
      }
    ]
  },
  {
    "id": "000000380665",
    "image": "coco/train2017/000000380665.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat kind of impact might the large red bus have on the rural community where it is parked?"
      },
      {
        "from": "gpt",
        "value": "The large red bus parked in front of a farmhouse may provide essential transportation services to the rural community. This could have a positive impact on the residents by connecting them to nearby urban areas, facilitating access to job opportunities, education, and other essential services. It may also promote tourism in the area, with visitors coming to explore the rural landscape and local attractions, ultimately benefiting local businesses and the economy. On the other hand, the presence of a large bus in a rural setting might cause minor traffic congestion or noise pollution in the otherwise quiet and peaceful environment, which could be a source of inconvenience to the residents."
      }
    ]
  },
  {
    "id": "000000527717",
    "image": "coco/train2017/000000527717.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What can be inferred about the purpose of this doll's outfit and accessory?\n<image>"
      },
      {
        "from": "gpt",
        "value": "The doll is dressed in a bunny or rabbit costume and holding a fake carrot, implying that the outfit and accessory aim to create a thematic representation of the bunny character. The combination of the animal costume and the carrot suggests that the doll is designed or dressed for a specific theme, event, or role-play. The costume and the carrot accessory together serve to emphasize the rabbit character, making it appear more playful and appealing, particularly to children or collectors interested in themed dolls or animal-inspired toys."
      }
    ]
  }
]

conv_template = "llava_llada" 
image_folder = "/group/40033/public_datasets"
for item in data_list:
    img_path = os.path.join(image_folder, item['image'])
    shutil.copy(img_path, './')
    image = Image.open(img_path)
    image_tensor = process_images([image], image_processor, model.config)
    print('process image, img num:', len(image_tensor), image_tensor[0].shape)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    question = item['conversations'][0]['value']
    answer = item['conversations'][1]['value']

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print(prompt_question)

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]
    
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        revise=False,
        steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
    )

    # print(cont)
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print(question)
    print(text_outputs)
