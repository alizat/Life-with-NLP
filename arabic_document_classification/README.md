# Multi-class Classification of Arabic Documents
In this project, I attempt to predict document type out of 5 different types for > 100K Arabic texts. The texts and their labels were obtained from [here](https://data.mendeley.com/datasets/v524p5dhpj/2).

## What's done: 
I generated static embeddings with an NLP model, [AraEuroBert-210M_distilled](https://huggingface.co/Abdelkareem/AraEuroBert-210M_distilled) and use them to train some standard ML classifiers:
- Logistic Regression
- Random Forest
- XGBoost
- Fine-tuned the above-mentioned SentenceTransformer model (model2vec-style)  -->  still needs a fix

I also fine-tuned the Hugging Face Transformer model, [AraEuroBert-210M](https://huggingface.co/Omartificial-Intelligence-Space/AraEuroBert-210M), which is the original model from which above-mentioned one was distilled. 

## Future Work
Here in this README, I will add a report of the classification results of the different models as soon as I can! Stay tuned.
