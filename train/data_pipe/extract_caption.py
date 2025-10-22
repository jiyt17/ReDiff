import json
from tqdm import tqdm
import numpy as np
import datasets
from PIL import Image
import io

# data_list = json.load(open('llava-1.5-org.json', 'r'))
# caption_list = []
# word_num = []
# caption_keys = ['What do you think is going on', 'happening', 'What are the key elements in this picture', 'Describe', 'What is this photo about', 'Analyze the image', 'description', 'Explain the visual content']
# for item in tqdm(data_list):
#     for key in caption_keys:
#         if key in item['conversations'][0]['value']:
#             caption_list.append(item)
#             word_num.append(len(item['conversations'][1]['value'].split()))
#             break
# print('caption_list:', len(caption_list))
# print('word num average:', np.mean(word_num))

# with open('llava-1.5-caption.json', 'w') as f:
#     json.dump(caption_list, f, indent=4)

# word_num = []
# data_list = json.load(open('/group/40005/public_datasets/ViCrit-Train/vicrit_train_equaltokens.json', 'r'))
# for item in tqdm(data_list):
#     word_num.append(len(item['conversations'][1]['value'].split()))
# print('word num average:', np.mean(word_num))

# subsets = ['llava', 'coco', 'sam', 'knowledge']
# for set in subsets:
    # dataset = datasets.load_dataset(f'/group/40005/public_datasets/ShareGPT4v/sharegpt4v_{set}/sharegpt4v({set})')
    # json_root = "/group/40005/public_datasets/ShareGPT4v/jsons"
    # img_root = "/group/40005/public_datasets/ShareGPT4v/images"
    # print('dataset len:', len(dataset['train']))
    # # word_num = []
    # res = []

    # for item in tqdm(dataset['train']):
    #     img = item['image']
    #     conv = item['conversations']
    #     id = item['id']
    #     img_name = f"{set}/{id}.jpg"
    #     img_path = f"{img_root}/{img_name}"
    #     try:
    #         img.save(img_path)
    #         res.append({'id': id, 'image': img_name, 'conversations': conv})
    #     except:
    #         continue
    #     # word_num.append(len(item['conversations'][1]['value'].split()))
    # # print('word num average:', np.mean(word_num))
    # print(set, 'len:', len(res))
    # with open(f"{json_root}/sharegpt4v_{set}.json", 'w') as f:
    #     json.dump(res, f, indent=4)


json_root = "/group/40005/public_datasets/ShareGPT4v/jsons/"
llava = json.load(open(json_root + 'sharegpt4v_llava.json', 'r'))
coco = json.load(open(json_root + 'sharegpt4v_coco.json', 'r'))[:45000]
sam = json.load(open(json_root + 'sharegpt4v_sam.json', 'r'))
knowledge = json.load(open(json_root + 'sharegpt4v_knowledge.json', 'r'))
sam_new = []
for item in llava:
    item['conversations'][1]['value'] = item['conversations'][1]['value'].replace('\n\n', '')
for item in coco:
    item['conversations'][1]['value'] = item['conversations'][1]['value'].replace('\n\n', '')
for item in sam:
    item['conversations'][1]['value'] = item['conversations'][1]['value'].replace('\n\n', '')
    if 'sa_' not in item['conversations'][1]['value']:
        sam_new.append(item)
for item in knowledge:
    item['conversations'][1]['value'] = item['conversations'][1]['value'].replace('\n\n', '')
final_res = llava + coco + sam_new + knowledge
print('final res len:', len(final_res))
for item in final_res:
    item['image'] = 'ShareGPT4v/images/' + item['image']
with open(json_root + 'sharegpt4v_offline.json', 'w') as f:
    json.dump(final_res, f, indent=4)



# vicrit = json.load(open('/group/40005/public_datasets/ViCrit-Train/vicrit_train_equaltokens.json', 'r'))
# print(len(vicrit))
# with open('/group/40005/public_datasets/ViCrit-Train/vicrit_train_equaltokens_offline.json', 'w') as f:
#     json.dump(vicrit, f, indent=4)

# res = json.load(open('/group/40005/public_datasets/ShareGPT4v/jsons/sharegpt4v_offline.json', 'r'))
# for item in res:
#     item['image'] = 'ShareGPT4v/images/' + item['image']
# with open('/group/40005/public_datasets/ShareGPT4v/jsons/sharegpt4v_offline.json', 'w') as f:
#     json.dump(res, f, indent=4)
