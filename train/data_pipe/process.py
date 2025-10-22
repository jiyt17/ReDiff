import json
import random
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from transformers import AutoTokenizer

# res = []
# anno = json.load(open('/group/40005/public_datasets/LLaVA-Instruct/llava_v1_5_mix665k.json', 'r'))
# for item in tqdm(anno):
#     if 'image' in item and len(item['conversations']) == 2:
#         if len(item['conversations'][1]['value'].split()) > 30:
#             res.append(item)

# print('len:', len(res))
# random.shuffle(res)
# with open('llava-1.5-org.json', 'w') as f:
#     json.dump(res, f, indent=4)


pretrained = "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_6"
model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)

anno1 = json.load(open('/group/40005/auroraji/LLaDA-V/train/data_pipe/o4_revise_base_s128_vicrit.json', 'r'))
anno2 = json.load(open('/group/40005/auroraji/LLaDA-V/train/data_pipe/o4_revise_base_s32_vicrit.json', 'r'))
anno = anno1 + anno2
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
    img = item['img'][len('/group/40005/public_datasets/'):]
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
with open('o4_revise_base_train.json', 'w') as f:
    json.dump(res, f, indent=4)
print('revise_total:', revise_total, 'revise_equal:', revise_equal, 'equal_ratio:', revise_equal / revise_total)




# res1 = json.load(open('/group/40005/auroraji/LLaDA-V/train/data_pipe/o4_revise_base_s128_vicrit_1.json'))
# res2 = json.load(open('/group/40005/auroraji/LLaDA-V/train/data_pipe/o4_revise_base_s128_vicrit_2.json'))
# res = res1 + res2
# print(len(res))
# with open('/group/40005/auroraji/LLaDA-V/train/data_pipe/o4_revise_base_s128_vicrit.json', 'w') as f:
#     json.dump(res, f, indent=4)