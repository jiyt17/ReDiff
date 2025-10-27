import transformers
import tokenizers
import json
from tqdm import tqdm
from datasets import load_dataset

tokenizer = transformers.AutoTokenizer.from_pretrained(
    'pretrained/LLaDA-V',
)


anno = json.load(open('ViCrit-Train/vicrit_train.json', 'r'))
print(len(anno))


res = []
for item in tqdm(anno):
    ori_text = item['original_caption']
    changed_text = item['changed_caption']
    ori_sol = item['solution_original']
    changed_sol = item['solution_target']
    
    ori_tokens = tokenizer.tokenize(ori_text)
    changed_tokens = tokenizer.tokenize(changed_text)
    ori_sol_tokens = tokenizer.tokenize(ori_sol)
    changed_sol_tokens = tokenizer.tokenize(changed_sol)
    if len(ori_tokens) == len(changed_tokens) and len(ori_sol_tokens) == len(changed_sol_tokens):
        
        conv_hal = [
            {
                "from": "human",
                "value": f"<image>\n{item['problem']}"
            },
            {
                "from": "gpt",
                "value": item['changed_caption']
            }
        ]
        revise = [{
            "org": changed_sol,
            "target": ori_sol
        }]
        new_item = {'image': item['image'].split('/')[-1], 'conversations': conv_hal, 'revise': revise}
        res.append(new_item)

with open('ViCrit-Train/vicrit_train_equaltokens.json', 'w') as f:
    json.dump(res, f, indent=4)
