from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import pandas as pd
from datasets import load_dataset
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from main import train_dataset, validation_dataset, test_dataset


checkpoint  = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

def tokenize_fn(batch):
  return tokenizer(batch['text'], truncation = True, padding = "max_length", max_length= 30)

train_tokenized_dataset = train_dataset.map(tokenize_fn, batched = True)
test_tokenized_dataset = validation_dataset.map(tokenize_fn, batched = True)
valid_tokenized_dataset = test_dataset.map(tokenize_fn, batched = True)

def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis = -1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average = 'micro')
  return {'accuracy': acc, 'f1_score': f1}

training_args = TrainingArguments(
  output_dir="./results",
  num_train_epochs= 5,
  per_device_train_batch_size = 32,
  per_device_eval_batch_size = 32,
  evaluation_strategy="epoch",
  save_strategy = 'epoch',
  logging_dir="./logs",
  logging_steps=10,
  load_best_model_at_end=True,
)
trainer = Trainer(
  model,
  training_args,
  train_dataset = train_tokenized_dataset,
  eval_dataset = valid_tokenized_dataset,
  tokenizer = tokenizer,
  compute_metrics = compute_metrics,
  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

test_output = trainer.evaluate(test_tokenized_dataset)
print("Test Accuracy:", test_output['eval_accuracy'], "Test Loss:", test_output['eval_loss'])