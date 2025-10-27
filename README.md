# From Denoising to Refining: A Corrective Framework for Vision-Language Diffusion Model
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2510.19871)
[![deploy](https://img.shields.io/badge/Hugging%20Face-LLaDA_V-FFEB3B)](https://huggingface.co/jiyatai/ReDiff)

  
## Introduction
We introduce ReDiff, a refining-enhanced vision-language diffusion model.

<img src="assets/teaser.jpg">


### Quick Inference Demo
The [ReDiff model](https://huggingface.co/jiyatai/ReDiff) is now available on Hugging Face Hub. To quickly test the model with a visual instruction demo, follow these simple steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/jiyt17/ReDiff
   cd train
   ```
2. **Initialize the environment**  
   Run the environment setup script to install necessary dependencies:
   ```bash
   bash init_env.sh
   ```
3. **Run the demo script**  
   Execute the demo script to test ReDiff on an example image:
   ```bash
   python generate_demo.py
   ```

## Training
Our model is trained from [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V).

### Foundational revision training
We train model to revise two types of synthetic errors: syntactic errors and semantic hallucinations. 

The syntactic errors are injected by randomly replacing a fraction of tokens with other tokens from the vocabulary, and semantic hallucinations are from ViCrit dataset, which provides pairs of correct captions and captions with factual errors.

1. Training data preparation:

   We select detailed image captioning as the representative task to validate our framework on enhancing the generative capabilities of vision-language diffusion models.

   The data sources contain [ViCrit](https://huggingface.co/datasets/zyang39/ViCrit-Train), [LLaVA-1.5](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), [ShareGPT4V](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), utilizing 160k, 20k and 80k data respectively.

   ```bash
   cd train/data_pipe
   python preprocess_vicrit.py
   ```

2. Training script:
   ```bash
   cd train
   bash ./scripts/llada_v_finetune_vicrit.sh
   ```

### Online self-correction learning

In the second stage, ReDiff-base generates its own flawed "drafts". These drafts, containing the model's grammatical and hallucinatory errors, are then revised by an expert AI assistant. 

1. Data praparation with o4-mini.
   ```bash
   cd train/data_pipe
   bash inference.sh
   python data_pipeline_gpt.py
   python process.py
   ```

   Data sample:
   ```python
   {
        "image": "ViCrit-Train/images/535.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a detailed description of the given image."
            },
            {
                "from": "gpt",
                "value": "In the image a smiling black man is standing outside a brick wall. He is holding a a gray magazine with a white cover on it. The man dressed in a gray suit suit and a white shirt shirt with a yellow tie. His tie tie is a gold color and has blue stripes.  He is a wearing a brown hat with a a white logo on it.  The magazine he is holding has a picture of on and it and a black background with a white text on it. The title \"the scene\" is also visible in the magazine. The man appears to be the featured on the front cover of the magazine."
            }
        ],
        "revise": [
            {
                "org": "a white shirt shirt",
                "target": "a light blue shirt"
            },
            {
                "org": "tie tie",
                "target": "gold tie"
            },
            {
                "org": "a picture of on and it",
                "target": "a man on the front cover"
            },
            {
                "org": "black background",
                "target": "red background"
            }
        ]
    }
   ```

2. Training script:
   ```bash
   cd train
   bash ./scripts/llada_v_finetune_o4.sh
   ```


## Evaluation
We tend to improve the generation quality of vision-language diffusion model, and demonstrate the effectiveness of refining-enhanced diffusion framework on three detailed image caption benchmarks: [CapMAS](https://github.com/adobe-research/CapMAS) (3 metrics: CLAIR for overall caption quality, Coverage for the comprehensiveness of the description, and Factuality for the accuracy of the content), [CapArena](https://github.com/njucckevin/CapArena) (score based on pairwise comparison) and [DetailCaps-4870](https://github.com/foundation-multimodal-models/CAPTURE) (metric: CAPTURE).

Evaluation script:
   ```bash
   cd eval
   bash inference.sh
   ```


## Acknowledgments
The code is largely based on the [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), training data source contains [ViCrit](https://huggingface.co/datasets/zyang39/ViCrit-Train), [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data). We thank the authors for their great work.

## Citation

```bibtex
@article{ji2025denoising,
  title={From Denoising to Refining: A Corrective Framework for Vision-Language Diffusion Model},
  author={Ji, Yatai and Wang, Teng and Ge, Yuying and Liu, Zhiheng and Yang, Sidi and Shan, Ying and Luo, Ping},
  journal={arXiv preprint arXiv:2510.19871},
  year={2025}
}
```

