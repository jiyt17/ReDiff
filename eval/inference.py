from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
import llava.utils_dist as dist

from PIL import Image
import requests
import copy
import torch
import shutil
import json
from tqdm import tqdm
import os
import time
import argparse

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = False  # In this demo, we consider using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it.

warnings.filterwarnings("ignore")

# res = json.load(open('model_outputs_llava-1.5.json', 'r'))
# new_res = []
# for item in res:
#     new_res += item
# print(len(new_res))
# with open('model_outputs_llava-1.5.json', 'w') as f:
#     json.dump(new_res, f, indent=4)


def eval_model(args):

    # Device setup
    dist.init()
    world_size, global_rank = dist.size(), dist.rank()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # pretrained = "/group/40034/auroraji/pretrained/LLaDA-V"
    pretrained = args.pretrained
    model_name = "llava_llada"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=devices[0])  # Add any other thing you want to pass in llava_model_args
    model.eval()

    if 'base' in pretrained:
        revise = False
    else:
        revise = True
    # revise = False

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

    # read data and split
    data_list = []
    for f in os.listdir(args.image_dir):
        data_list.append(os.path.join(args.image_dir, f))
    data_list = data_list[global_rank::world_size]

    conv_template = "llava_llada" 

    res = []
    for img_path in tqdm(data_list, total=len(data_list), disable=global_rank != 0):
        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        # print('process image, img num:', len(image_tensor), image_tensor[0].shape)
        image_tensor = [_image.to(dtype=torch.float16, device=devices[0]) for _image in image_tensor]

        question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        # print(prompt_question)

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(devices[0])
        image_sizes = [image.size]
        
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            revise=revise,
            revise_topk=args.revise_topk,
            revise_start=args.revise_start,
            steps=args.steps, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
        )

        # print(cont)
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)[0]
        # print(question)
        # print(text_outputs)

        img_id = img_path.split('/')[-1]
        res.append({'image_id': img_id, 'text': text_outputs})

    if dist.size() > 1:
        res = dist.all_gather(res)

    if dist.is_main():
        save_res = {}
        for res_single_gpu in res:
            for item in res_single_gpu:
                save_res[item['image_id']] = item['text']
        print(f"Saving results to {args.output}")
        # mkdir
        if not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))
        with open(args.output, 'w') as f:
            json.dump(save_res, f, indent=4)

if __name__ == "__main__":
    # read args
    parser = argparse.ArgumentParser(description="Evaluate LLaDA-V model on caption benchmarks")
    parser.add_argument("--pretrained", type=str, default="/group/40034/auroraji/pretrained/LLaDA-V", help="Path to the pretrained model")
    parser.add_argument("--output", type=str, default="/group/40034/auroraji/CapMAS/predictions", help="Path to save the model outputs")
    parser.add_argument("--image_dir", type=str, default="/group/40005/auroraji/CapArena/data/caparena_auto_docci_600", help="Directory containing images for evaluation")
    parser.add_argument("--steps", type=int, default=128, help="Number of steps for generation")
    parser.add_argument("--revise-topk", type=float, default=None, help="Revise topk")
    parser.add_argument("--revise-start", type=float, default=1.0, help="Revise start")
    args = parser.parse_args()
    eval_model(args) 