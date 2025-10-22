
import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np
import io

import webdataset as wds
import torch.distributed as dist

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from llava.conversation import conv_templates
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
# from llava.train.llava_trainer import LLaVATrainer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images

from llava import conversation as conversation_lib
from llava.model import *
# from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord, rank_print

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    prompt_chunks_tokens = [tokenizer.tokenize(chunk) for chunk in prompt.split("<image>")]
    print([len(pc) for pc in prompt_chunks])
    print([pc for pc in prompt_chunks])
    print([len(pc) for pc in prompt_chunks_tokens])
    print([pc for pc in prompt_chunks_tokens])
    tokens = []
    for pct in prompt_chunks_tokens:
        tokens += pct

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
        
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long), tokens
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids, tokens


prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = False


data_path = "/group/40005/public_datasets/LLaVA-Instruct/llava_v1_5_mix665k.json"
image_folder = "/group/40005/public_datasets"
# data_list = json.load(open(data_path, "r"))

# pretrained = "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_6"
pretrained = "/group/40005/auroraji/pretrained/LLaDA-V"

model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)
model.eval()

# item = data_list[0]
# item = {
#     "id": "000000261414",
#     "image": "coco/train2017/000000261414.jpg",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "What makes the image noteworthy?\n<image>"
#       },
#       {
#         "from": "gpt",
#         "value": "The image is noteworthy due to the unusual and amusing combination of elements involving the man surfing in the ocean. Instead of a typical surfing scene, the man is dressed in a wacky costume, decorated with souvenirs such as the Union Jack, and holding fishing rods. His appearance, along with him riding a surfboard with an Afro, makes the image stand out and captures the viewer's attention for its humor and unconventional aspects. It also highlights the creative and playful elements that can be found in edited images, pushing beyond the boundaries of ordinary surfing photos."
#       }
#     ]
#   }
item = {
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
  }
img_path = os.path.join(image_folder, item['image'])

image = Image.open(img_path)
image_tensor = process_images([image], image_processor, model.config)
print('process image, img num:', len(image_tensor))
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 
# question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
question = item['conversations'][0]['value']
answer = item['conversations'][1]['value']
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], answer)
# print(conv)
prompt_question = conv.get_prompt()
# print(prompt_question)
# delete extra tokens in the last positions
extra_len = len("<|start_header_id|>assistant<|end_header_id|>\n\n")
prompt_question = prompt_question[:-extra_len]
print(prompt_question)

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


input_ids, tokens = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
input_ids = input_ids.unsqueeze(0).to(device)
print(len(input_ids[0]), input_ids)
print(len(tokens))
tokens_ids = {}
for i in range(len(tokens)):
    tokens_ids[i] = tokens[i]
print(tokens_ids)
image_sizes = [image.size]

mask_id = 126336
mask_indices = [80,85,87,89,90,106,107,135,136]
for i in mask_indices:
    input_ids[0, i] = mask_id
ans_len = 89

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    revise=False,
    steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
)
# attn = cont[-1]
# print(len(attn))
# attn = torch.cat(attn)
# print(attn.shape)
# ans_scores = attn[4].mean(1).sum(-2).squeeze()[-ans_len:].cpu()
# for token, score in zip(tokens[-ans_len:], ans_scores.tolist()):
    # print(token, score)

# topk_index = torch.topk(ans_scores, 10).indices
# for i in topk_index:
    # print(tokens[-ans_len:][i])

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)[0]
print(text_outputs)
text_outputs = tokenizer.tokenize(text_outputs)
for i in mask_indices:
    print(tokens_ids[i], '->', text_outputs[i-(len(tokens)-ans_len)])

# def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
#     is_multimodal = data_args.is_multimodal
#     if not is_multimodal:
#         return sources

#     for source in sources:
#         for sentence in source:
#             # TODO maybe this should be changed for interleaved data?
#             # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
#             # only check for num_im=1
#             num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
#             if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
#                 sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
#                 sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
#                 sentence["value"] = sentence["value"].strip()
#                 if "mmtag" in conversation_lib.default_conversation.version:
#                     sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
#             replace_token = DEFAULT_IMAGE_TOKEN
#             if data_args.mm_use_im_start_end:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#             # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
#             sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

#     return sources


# def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
#     """
#     Given a list of sources, each is a conversation list. This transform:
#     1. Add signal '### ' at the beginning each sentence, with end signal '\n';
#     2. Concatenate conversations together;
#     3. Tokenize the concatenated conversation;
#     4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
#     """
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
#         return preprocess_plain(sources, tokenizer)
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
#         return preprocess_llama_2(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "llava_llada":
#         return preprocess_llada(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "llada_plain":
#         return preprocess_plain(sources, tokenizer)
#     if conversation_lib.default_conversation.version.startswith("v1"):
#         return preprocess_v1(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "mpt":
#         return preprocess_mpt(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "qwen":
#         return preprocess_qwen(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "gemma":
#         return preprocess_gemma(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "llama_v3":
#         return preprocess_llama3(sources, tokenizer, has_image=has_image)
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         header = f"{conversation_lib.default_conversation.system}\n\n"
#         conversation = _add_speaker_and_signal(header, source)
#         conversations.append(conversation)

#     # tokenize conversations
#     def get_tokenize_len(prompts):
#         return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

#     if has_image:
#         input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
#     else:
#         conversations_tokenized = _tokenize_fn(conversations, tokenizer)
#         input_ids = conversations_tokenized["input_ids"]

#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         if has_image:
#             tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
#         else:
#             tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
#         speakers = [sentence["from"] for sentence in source]
#         _mask_targets(target, tokenized_lens, speakers)

#     return dict(input_ids=input_ids, labels=targets)


# class source_dataset(Dataset):
#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.list_data_dict = []
#         with open(data_path, "r") as file:
#             cur_data_dict = json.load(file)
#             rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
#             self.list_data_dict.extend(cur_data_dict)
        
#         self.tokenizer = tokenizer
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)
    
#     def process_image(self, image_file, overwrite_image_aspect_ratio=None):
#         image_folder = self.data_args.image_folder
#         image_folder_2 = getattr(self.data_args, 'image_folder_2', None)
#         processor = self.data_args.image_processor

#         image_path = None
#         if image_folder:
#             primary_path = os.path.join(image_folder, image_file)
#             if os.path.exists(primary_path):
#                 image_path = primary_path
#             elif image_folder_2:
#                 secondary_path = os.path.join(image_folder_2, image_file)
#                 if os.path.exists(secondary_path):
#                     image_path = secondary_path
#             else:
#                 image_path = primary_path
                    
#         elif image_folder_2: # Only secondary folder is provided
#              image_path = os.path.join(image_folder_2, image_file)

#         try:
#             # image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
#             image = Image.open(image_path).convert("RGB")
#         except Exception as exn:
#             print(f"Failed to open image {image_path} (derived from {image_file}). Exception:", exn)
#             raise exn

#         image_size = image.size
#         image_aspect_ratio = self.data_args.image_aspect_ratio
#         if overwrite_image_aspect_ratio is not None:
#             image_aspect_ratio = overwrite_image_aspect_ratio
#         if image_aspect_ratio == "highres":
#             image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
#         elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
#             image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
#         elif image_aspect_ratio == "crop_split":
#             image = process_highres_image_crop_split(image, self.data_args)
#         elif image_aspect_ratio == "pad":

#             def expand2square(pil_img, background_color):
#                 width, height = pil_img.size
#                 if width == height:
#                     return pil_img
#                 elif width > height:
#                     result = Image.new(pil_img.mode, (width, width), background_color)
#                     result.paste(pil_img, (0, (width - height) // 2))
#                     return result
#                 else:
#                     result = Image.new(pil_img.mode, (height, height), background_color)
#                     result.paste(pil_img, ((height - width) // 2, 0))
#                     return result

#             image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
#             image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
#         else:
#             image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
#         return image, image_size, "image"
    
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         # TODO: define number of retries somewhere else
#         num_base_retries = 5
#         num_final_retries = 300

#         # try the current sample first
#         for attempt_idx in range(num_base_retries):
#             try:
#                 sample = self._get_item(i)
#                 return sample
#             except Exception as e:
#                 # sleep 1s in case it is a cloud disk issue
#                 print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
#                 time.sleep(1)

#         # try other samples, in case it is file corruption issue
#         for attempt_idx in range(num_base_retries):
#             try:
#                 next_index = min(i + 1, len(self.list_data_dict) - 1)
#                 # sample_idx = random.choice(range(len(self)))
#                 sample = self._get_item(next_index)
#                 return sample
#             except Exception as e:
#                 # no need to sleep
#                 print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
#                 pass

#         try:
#             sample = self._get_item(i)
#             return sample
#         except Exception as e:
#             raise e

#     def _get_item(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

#         if "image" in sources[0]:
#             image_file = self.list_data_dict[i]["image"]
#             if type(image_file) is list:
#                 image = [self.process_image(f) for f in image_file]
#                 # Handling multi images
#                 # overwrite to process with simple pad 
#                 if len(image_file) > 1:
#                     image = [self.process_image(f, "pad") for f in image_file]
#                     image = [[im[0], im[1], "image"] for im in image]
#             else:
#                 image = [self.process_image(image_file)]
#             org_sources = copy.deepcopy(sources)
#             sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])

#         has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
#         data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
#         # rank0_print('org:', data_dict['input_ids'][0].tolist(), data_dict['labels'][0].tolist())

#         if "prompt" in data_dict:
#             prompt = data_dict["prompt"]
#         else:
#             prompt = None

#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

#         # image exist in the data
#         if "image" in self.list_data_dict[i]:
#             data_dict["image"] = image
#         elif "video" in self.list_data_dict[i]:
#             data_dict["image"] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict["image"] = [
#                 (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
#             ]
#         # prompt exist in the data
#         if prompt is not None:
#             data_dict["prompt"] = prompt

#         data_dict["id"] = self.list_data_dict[i].get("id", i)
#         if conversation_lib.default_conversation.version == "llada_plain":
#             data_dict["is_plain"] = True
#             data_dict["is_llada"] = True
#         elif conversation_lib.default_conversation.version == "llava_llada":
#             data_dict["is_plain"] = False
#             data_dict["is_llada"] = True

#         if hal_version:
#             data_dict_hal = preprocess(sources_hal, self.tokenizer, has_image=has_image)
#             # rank0_print('hal:', data_dict_hal['input_ids'][0].tolist(), data_dict_hal['labels'][0].tolist())
#             # rank0_print(data_dict["input_ids"][0].shape, data_dict_hal["input_ids"][0].shape, data_dict["labels"][0].shape)
#             hal_sol = self.tokenizer.encode('dummy ' + org_sources[0]['ori_sol'])
#             # hal_sol_tokens = self.tokenizer.tokenize('dummy ' + org_sources[0]['ori_sol'])
#             # rank0_print('hal_org:', len(hal_sol), hal_sol[1:], org_sources[0]['ori_sol'])
#             hal_sol = hal_sol[1:]
#             hal_st = -1
#             # revise_mask1 = data_dict["input_ids"][0] != data_dict_hal["input_ids"][0]
#             revise_indices = torch.zeros_like(data_dict["input_ids"], dtype=torch.bool)
#             for j in range(len(data_dict["input_ids"])):
#                 if data_dict["input_ids"][j] == hal_sol[0]:
#                     match = True
#                     for t in range(len(hal_sol)):
#                         if data_dict["input_ids"][j+t] != hal_sol[t]:
#                             match = False
#                             break
#                     if match:
#                         hal_st = j
#                         break
#             if hal_st > 0:
#                 revise_indices[hal_st: hal_st+len(hal_sol)] = True
#             # rank0_print(revise_mask1, revise_mask2)
#             # rank0_print(data_dict['input_ids'][revise_indices], data_dict['labels'][revise_indices])

#             data_dict['input_ids'] = data_dict_hal['input_ids'][0]
#             data_dict['revise_indices'] = revise_indices

#         return data_dict