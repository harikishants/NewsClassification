import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random

from dataset import BBCDataset
from tokenizer import BPETokenizer
from model import BERTClassifier
from train import train_model

 
df = pd.read_csv("archive/bbc-news-data.csv", sep='\t')

texts = df['content'].tolist()
categories = df['category'].tolist()
labels = []
for category in categories:
    if category == 'business':
        labels.append(0)
    elif category == 'entertainment':
        labels.append(1)
    elif category == 'politics':
        labels.append(2)
    elif category == 'sport':
        labels.append(3)
    elif category == 'tech':
        labels.append(4)
    else:
        print(f'[error] Unknown catgeory: {category}')

tokenizer = BPETokenizer(vocab_file='vocab/vocab_final')

# X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(random.sample(texts, 10), random.sample(labels, 10), test_size=0.1)

train_dataset = BBCDataset(X_train, y_train, tokenizer)
val_dataset = BBCDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = BERTClassifier(vocab_size = len(tokenizer.vocab))

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=5, device=device)
