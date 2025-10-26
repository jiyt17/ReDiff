import json
import random
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from transformers import AutoTokenizer



pretrained = "pretrained_model_path"
model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)

anno = json.load(open('o4_revise_rediff_s128_vicrit.json', 'r'))
print('anno len:', len(anno))

id = 0
res = []
revise_total = 0
revise_equal = 0
for item in tqdm(anno):
    revise = item['revise']
    if revise == 'right':
        continue
    try:
        revise = revise.replace('```', '').replace('json', '')
        revise = json.loads(revise)
        # print(revise)
    except:
        print(revise)
        continue
    img = item['img']
    conv = [
        {
            "from": "human",
            "value": item['ques']
        },
        {
            "from": "gpt",
            "value": item['pred']
        }
    ]
    new_revise = []
    for pair in revise:
        revise_total += 1
        try:
            org = tokenizer.tokenize(pair['org'])
            target = tokenizer.tokenize(pair['target'])
        except:
            print(pair)
            continue
        if len(org) == len(target):
            revise_equal += 1
            new_revise.append(pair)
    if len(new_revise) > 0:
        res.append({'id': id, 'image': img, 'conversations': conv, 'revise': new_revise})
        id += 1

print(id)
with open('o4_revise_train.json', 'w') as f:
    json.dump(res, f, indent=4)
print('revise_total:', revise_total, 'revise_equal:', revise_equal, 'equal_ratio:', revise_equal / revise_total)

