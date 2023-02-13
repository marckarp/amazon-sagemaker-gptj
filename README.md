# Fine-tune and deploy GPT-J-6B on Amazon SageMaker. 

In `GPTJ-SageMaker` folder this repo contains two notebooks:
1. Fine-Tune-GPT-J.ipynb
2. Deploy-GPT-J.ipynb

Use `Fine-Tune-GPT-J.ipynb` to fine-tune GPT-J-6B on the [quotes](https://www.kaggle.com/datasets/akmittal/quotes-dataset) dataset and `Deploy-GPT-J.ipynb` to deploy the model onto a SageMaker real-time endpoint.

`Finetune_GPTNEO_GPTJ6B` are artifacts from this repo: https://github.com/mallorbc/Finetune_GPTNEO_GPTJ6B

The model is fine-tuned on the task of generating quotes for specific catagories:
```
love: <AIGeneratedQuote>
```
