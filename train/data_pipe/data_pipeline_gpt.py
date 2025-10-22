import openai
import os
import base64
import requests
from tqdm import tqdm
import argparse
import random
import json
from sys_prompts import revise_prompt

# api_key="Ys3vC7zhfJHofZjEovqc2AIO@3695"
api_key="7EPX8pQ3dWlGD5ACnlc9qX0L@3695"
# api_key="JthCt5ls1kP2M7HlXOtIrIWu@420"
os.environ["OPENAI_API_KEY"]=api_key
os.environ["OPENAI_API_BASE"]="http://v2.open.venus.oa.com/llmproxy"
API_URL = "http://v2.open.venus.oa.com/llmproxy/chat/completions"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def process_image(prompt, image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    # "model": "gpt-4o-2024-08-06",
    # "model": "o3",
    "model": "o4-mini",
    # "model": "gemini-2.0-flash",
    # "model": "gemini-2.5-pro",
    "messages": [
        {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": revise_prompt
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
            {
                "type": "text",
                "text": f"{prompt}"
            }
        ]
        }
    ],
    # "max_tokens": 2000
    "max_completion_tokens": 8000
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    # print('response', response.json())
    try:
        content = response.json()['choices'][0]['message']['content']
    except:
        print(response)
        content = None
    return content


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str, default="Hi")
    # args = parser.parse_args()
    # response = process_text(args.prompt)
    # print(response)

    # case 1
    # img = "000000527717.jpg"
    # ques = "What can be inferred about the purpose of this doll's outfit and accessory?"
    # ans = "The doll is dressed in a bunny or rabbit costume and holding a fake carrot, implying that the outfit and accessory aim to create a thematic representation of the bunny character. The combination of the animal costume and the carrot suggests that the doll is designed or dressed for a specific theme, event, or role-play. The costume and the carrot accessory together serve to emphasize the rabbit character, making it appear more playful and appealing, particularly to children or collectors interested in themed dolls or animal-inspired toys."
    # pred = "The doll's outfit and accessory suggest that it is designed for imaginative play or display. The doll is dressed in a gray rabbit suit with a white hood and a white chest, complete with rabbit ears. The presence of a carrot as its accessory further adds to the imaginative and playful theme. The doll's placement against a colorful andwn rug indicates that it is meant to be a decorative point in the room. The doll itself, which resembles the rabbit and is positioned at the bottom of the doll, complements both the outfit and the carrot. Overall, the doll's design, outfit, and accessory contribute to creating a whimsical and imaginative display."
    # case 2
    # img = "000000261414.jpg"
    # ques = "What makes the image noteworthy?"
    # ans = "The image is noteworthy due to the unusual and amusing combination of elements involving the man surfing in the ocean. Instead of a typical surfing scene, the man is dressed in a wacky costume, decorated with souvenirs such as the Union Jack, and holding fishing rods. His appearance, along with him riding a surfboard with an Afro, makes the image stand out and captures the viewer's attention for its humor and unconventional aspects. It also highlights the creative and playful elements that can be found in edited images, pushing beyond the boundaries of ordinary surfing photos."
    # pred = "The image is noteworthy due to its striking depiction of a man in a colorful wetsuit, complete with a helmet, goggles, and a life vest, engaging in an exhilarating moment of kiteboarding. The wetsuit features a colorful British flag design, adding a splash of vibrancy to the scene. The man is skillfully maneuvering a kite, which is connected to a harness and attached to his lines, all as he rides across the surface of the water. The photograph is set in a tropical setting, with a white sandy beach and palm trees visible in the background, adding to the dynamic and vibrant nature of the scene."
    # case 3
    # img = "000000312621.jpg"
    # ques = "What would you consider to be the most notable feature of the building's exterior, and why?"
    # ans = "The most notable feature of the building's exterior is the large, red and white clock mounted on the side of the building above a high arch. The clock's prominent size and striking color make it an eye-catching focal point. Additionally, it is unique due to its location above a door and the presence of statues surrounding the clock. This design contributes to the building's aesthetic appeal and helps it stand out, making it easily recognizable and memorable."
    # pred = "The most notable feature of the building's exterior is the large red and white clock mounted on the wall. The clock, with its striking red frame and white numerals, stands out against the tan-colored brick wall. Its prominent placement and classic design make it the focal point of the building's exterior. The clock is both a functional and decorative element, providing the practical purpose of telling the time while also adding visual embellishment that enhances the overall appearance of the building. Outside of the clock, there are also several statues of people and a bust positioned at the top of the clock, adding to the historical and aesthetic significance of the building."
    
    
    # preds = json.load(open('model_finetune_15_s32_outputs_coco.json', 'r'))
    preds = json.load(open('llada_v_s32_outputs_vicrit_others.json', 'r'))
    res = []
    print(len(preds))
    # preds = preds[:2500]
    # preds = preds[1700:3400]
    preds = preds[2500:]

    if os.path.exists('o4_revise_base_s32_vicrit_2.json'):
       res = json.load(open('o4_revise_base_s32_vicrit_2.json', 'r'))
    start = len(res)

    for item in tqdm(preds[start:]):
        ques = item['ques'].replace('<image>', '').strip()
        ans = item['ans']
        pred = item['pred']
        img = item['img']

        prompt = f"Question: {ques}\nAnswer: {ans}\nPrediction: {pred}"
        response = process_image(prompt, img)
        print(response)

        item['revise'] = response
        res.append(item)
        # avoid disk failure
        try:
            with open('o4_revise_base_s32_vicrit_2_fake.json', 'w') as f:
                json.dump(res, f, indent=4)
        except:
            break
        with open('o4_revise_base_s32_vicrit_2.json', 'w') as f:
            json.dump(res, f, indent=4)