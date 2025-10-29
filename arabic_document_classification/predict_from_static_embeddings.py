"""
Before running this file, make sure you download the data from this link and unzip it next to this script:
https://data.mendeley.com/datasets/v524p5dhpj/2
"""

import pandas as pd
import numpy as np
import time

import torch
# from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
data_sampled = data.sample(frac=1, random_state=42).reset_index(drop=True)      # Shuffle the dataset
# data_sampled = data.sample(frac=0.1, random_state=42).reset_index(drop=True)  # 1% of the dataset


# Display dataset info
print(f"  Number of documents: {len(data_sampled)}")
print("  Class distribution:  (0: culture, 1: diverse, 2: economy, 3: politic, 4: sport)")
print(data_sampled.value_counts("label"))
# data.value_counts("label").sort_index().plot(kind="bar", title="Class Distribution")


# Get embeddings (feature matrix)
print("\nExtracting embeddings...")
start_time = time.time()
print(f"  Using model: {model_name}")
extractor = SentenceTransformer(model_name, trust_remote_code=True, device=device)
embeddings = extractor.encode(data_sampled["text"].tolist())
print(f"  Embeddings shape: {embeddings.shape}")
end_time = time.time()
print(f"  Time elapsed: {end_time - start_time:.2f} seconds")


# Train-validation-test split (70-15-15)
print("\nSplitting dataset into train, validation and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(embeddings, data_sampled["label"].tolist(), test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"  Training set size: {len(X_train)}")
print(f"  Validation set size: {len(X_valid)}")
print(f"  Test set size: {len(X_test)}")


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
model = xgb.train(
    params,
    dtrain,
    num_boost_round = 2000,
    evals = [(dtrain, "train"), (dvalid, "validation")],
    early_stopping_rounds=20,
    verbose_eval=100
)
y_pred_prob = model.predict(xgb.DMatrix(X_test))
y_pred = np.argmax(y_pred_prob, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# print('\n=============================================')
# print('\n*** Document Classification with Fine-tuned Model ***')
# print('\n-------')
