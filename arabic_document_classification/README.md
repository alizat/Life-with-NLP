# Multi-class Classification of Arabic Documents
In this project, I attempt to predict document type out of 5 different types for > 100K Arabic texts. The texts and their labels were obtained from [here](https://data.mendeley.com/datasets/v524p5dhpj/2).

## What's done so far:  
I generated static embeddings with an NLP model, [AraEuroBert-210M_distilled](https://huggingface.co/Abdelkareem/AraEuroBert-210M_distilled) and use them to train some standard ML classifiers:
- Logistic Regression
- Random Forest
- XGBoost

## Future Work
I intend to fine-tune the above-mentioned NLP model on the data to improve upon the prediction performance that was achieved with the static embeddings. Stay tuned!