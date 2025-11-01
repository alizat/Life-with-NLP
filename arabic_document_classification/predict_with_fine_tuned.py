"""
Fine-tuning Abdelkareem/AraEuroBert-210M_distilled on the Arabic classification dataset.
Before running this file:
    1. Download and unzip the dataset next to this script:
       https://data.mendeley.com/datasets/v524p5dhpj/2
"""

import os
import pandas as pd
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

# -------------------
# Setup
# -------------------
print("\nSetup...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
model_name = "Omartificial-Intelligence-Space/AraEuroBert-210M"

# load previously trained model if exists
model_dir = "./fine_tuned_arabert_model"  # where a saved model would exist is this script was run previously
if os.path.isdir(model_dir):
    model_name = model_dir
print(f"Using model: {model_name}")

# -------------------
# Load Dataset
# -------------------
print("\nLoading dataset...")
data = pd.read_csv("arabic_dataset_classifiction.csv", encoding="utf-8")
data.columns = ["text", "label"]
data = data[data['text'].notna()].reset_index(drop=True)

print("\nSampling dataset...")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

num_labels = len(data["label"].unique())
print(f"  Number of samples: {len(data)}")
print(f"  Number of labels: {num_labels}")
print("  Label distribution:")
print(data["label"].value_counts())

# -------------------
# Split Data
# -------------------
train_df, temp_df = train_test_split(data,    test_size=0.3, random_state=42)  # , stratify=data["label"]
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # , stratify=temp_df["label"]

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
test_ds  = Dataset.from_pandas(test_df)

# -------------------
# Tokenization
# -------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=256)#.to(device)

print("Tokenizing dataset...")
train_ds = train_ds.map(preprocess_function, batched=True)
valid_ds = valid_ds.map(preprocess_function, batched=True)
test_ds  = test_ds.map(preprocess_function, batched=True)

# -------------------
# Load Model
# -------------------
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    trust_remote_code=True
).to(device)

# -------------------
# Training Setup
# -------------------
# # Colab T4 GPU
# training_args = TrainingArguments(
#     output_dir="./fine_tuned_arabert",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     logging_dir="./logs",
#     logging_steps=50,
# )

# 16 GB RAM + NVIDIA RTX 3070
training_args = TrainingArguments(
    output_dir="./fine_tuned_arabert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,                     # ✅ good for BERT-like models
    per_device_train_batch_size=8,          # 🔽 reduce from 16 → 8 for 8 GB VRAM
    per_device_eval_batch_size=8,           # match training batch size
    num_train_epochs=3,                     # start smaller; can increase later
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,                     # keep only best + latest checkpoint
    fp16=True,                              # ✅ mixed precision → saves VRAM, faster on RTX GPUs
    gradient_accumulation_steps=2,          # ✅ simulates batch size 16 effectively
    warmup_ratio=0.1,                       # helps stabilize early training
    report_to="none",                       # or "tensorboard" if you want logs
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------
# Metrics Function
# -------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# -------------------
# Trainer
# -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------
# Train Model
# -------------------

# if NO previously trained model exists...
if not os.path.isdir(model_dir):
    print("\nTraining model...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

# -------------------
# Evaluate on Test Set
# -------------------
print("\nEvaluating on test set...")
predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
print(f"Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(labels, preds))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))

# -------------------
# Save Fine-Tuned Model
# -------------------
# if NO previously trained model exists...
if not os.path.isdir(model_dir):
    trainer.save_model("./fine_tuned_arabert_model")
    tokenizer.save_pretrained("./fine_tuned_arabert_model")
    print("\nModel saved to ./fine_tuned_arabert_model")
