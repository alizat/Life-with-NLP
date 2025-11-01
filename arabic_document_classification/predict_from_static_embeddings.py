"""
Before running this file, make sure you download the data from this link and unzip it next to this script:
https://data.mendeley.com/datasets/v524p5dhpj/2
"""

import pandas as pd
import numpy as np
import time
import pickle

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb


# cuda
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"  
    torch.cuda.empty_cache()


# Parameters
# model_name = "Omartificial-Intelligence-Space/AraEuroBert-210M"
model_name = "Abdelkareem/AraEuroBert-210M_distilled"


# Load dataset
print("\nLoading dataset...")
data = pd.read_csv("arabic_dataset_classifiction.csv", encoding="utf-8")
data.columns = ["text", "label"]
data = data[data['text'].notna()]


# Sample dataset for quicker experimentation
print("\nSampling dataset...")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)      # Shuffle the dataset
# data = data.sample(frac=0.1, random_state=42).reset_index(drop=True)  # 10% of the dataset


# Display dataset info
print(f"  Number of documents: {len(data)}")
print("  Class distribution:  (0: culture, 1: diverse, 2: economy, 3: politic, 4: sport)")
print(data.value_counts("label"))
# data.value_counts("label").sort_index().plot(kind="bar", title="Class Distribution")


# Train-validation-test split (70-15-15)
print("\nSplitting dataset into train, validation and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(data["text"].tolist(), data["label"].tolist(), test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp,                y_temp,                 test_size=0.5, random_state=42)
print(f"  Training set size:   {len(X_train)}")
print(f"  Validation set size: {len(X_valid)}")
print(f"  Test set size:       {len(X_test)}")


# keep a copy of the texts before obtaining embeddings
X_train_base = X_train.copy()
X_valid_base = X_valid.copy()
X_test_base  = X_test.copy()


# Get embeddings (feature matrix)
print("\nExtracting embeddings...")
start_time = time.time()
print(f"  Using model: {model_name}")
extractor = SentenceTransformer(model_name, trust_remote_code=True, device=device)
X_train = extractor.encode(X_train)
X_valid = extractor.encode(X_valid)
X_test  = extractor.encode(X_test)
print(f"  Embeddings shape (train): {X_train.shape}")
end_time = time.time()
print(f"  Time elapsed: {end_time - start_time:.2f} seconds")


print('\n=============================================')
print('\n*** Document Classification with Static Embeddings ***')
print('\n-------')


# Train a and evaluate a Logistic Regression classifier
print("\nTraining Logistic Regression classifier...")
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)
print("  Logistic Regression classifier trained.")
y_pred_lr = lr_classifier.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy: {accuracy_lr * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print('\n-------')


# # Train and evaluate a Random Forest classifier
# print("\nTraining Random Forest classifier...")
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)
# print("  Random Forest classifier trained.")
# y_pred_rf = rf_classifier.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_rf))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_rf))
# print('\n-------')


# Train and evaluate an XGBoost classifier
# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html#setting-parameters
print("\nTraining an XGBoost classifier...")
params = {
    "objective": "multi:softprob",  # "multi:softmax"
    "num_class": 5,
    "eval_metric": "mlogloss",  # "merror"
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}
print('Params:', params)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest  = xgb.DMatrix(X_test,  label=y_test)
xgb_classifier = xgb.train(
    params,
    dtrain,
    num_boost_round = 2000,
    evals = [(dtrain, "train"), (dvalid, "validation")],
    early_stopping_rounds=20,
    verbose_eval=100
)
y_pred_prob = xgb_classifier.predict(xgb.DMatrix(X_test))
y_pred_xgb = np.argmax(y_pred_prob, axis=1)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"  Accuracy: {accuracy_xgb * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print('\n-------')


# print('\n=============================================')
# print('\n*** Document Classification with Fine-tuned SentenceTransformer (model2vec) Model ***')
# print('\n-------')


# print("\nFine-tuning SentenceTransformer (model2vec-style model)...")
# model = extractor
# train_examples = [InputExample(texts=[x, x], label=y) for x, y in zip(X_train_base, y_train)]  # x: str, y: int
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.SoftmaxLoss(  # classification-compatible loss (e.g., SoftmaxLoss)
#     model=model,
#     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
#     num_labels=len(set(y_train))
# )
# model.fit(  # fine-tune
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=3,
#     warmup_steps=100,
#     output_path="./fine_tuned_sentence_transformer"  # save model
# )
# with open('fine_tuned_sentence_transformer/softmax_loss.pkl', 'wb') as file:
#     pickle.dump(train_loss, file)  # save classification head


# # load saved model and classification head
# model_path = "./fine_tuned_sentence_transformer"
# model = SentenceTransformer(model_path)
# model.eval()
# with open('fine_tuned_sentence_transformer/softmax_loss.pkl', 'rb') as file:
#     train_loss = pickle.load(file)


# # predict
# embeddings = model.encode(X_test_base, convert_to_tensor=True, show_progress_bar=True)  # get embeddings
# with torch.no_grad():
#     logits = train_loss.classifier(embeddings)     # shape: [N, num_labels]
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     y_preds_ftst = torch.argmax(probs, dim=-1).cpu().numpy()
# accuracy_xgb = accuracy_score(y_test, y_preds_ftst)
# print(f"  Accuracy: {accuracy_xgb * 100:.2f}%")
# print("\nClassification Report:")
# print(classification_report(y_test, y_preds_ftst))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_preds_ftst))
# print('\n-------')
