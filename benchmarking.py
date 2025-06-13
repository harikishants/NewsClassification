import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

from dataset import BBCDatasetBM
from train import train_model

if __name__ == "__main__": # for benchmarking custom built BERT

    df = pd.read_csv("data/bbc-news-data.csv", sep='\t')

    texts = df['content'].tolist()
    categories = df['category'].tolist()
    label_map = {
        'business': 0,
        'entertainment': 1,
        'politics': 2,
        'sport': 3,
        'tech': 4
    }
    labels = [label_map[c] for c in categories]

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # np.random.seed(42)
    # indices = np.random.choice(len(texts), 10, replace=False)
    # sampled_texts = [texts[i] for i in indices]
    # sampled_labels = [labels[i] for i in indices]

    # X_train, X_val, y_train, y_val = train_test_split(sampled_texts, sampled_labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = BBCDatasetBM(X_train, y_train, tokenizer)
    val_dataset = BBCDatasetBM(X_val, y_val, tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    for param in model.parameters(): # to make all parameters trainable ie: unfreeze
        param.requires_grad = True

    # optimizer = AdamW(model.parameters(), lr=2e-4)

    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},         # Pretrained BERT gets a low LR
        {'params': model.classifier.parameters(), 'lr': 1e-3}    # Classifier head gets a higher LR
    ])

    num_epochs = 20
    num_training_steps = len(train_loader) * num_epochs

    # gradually reduction of LR
    scheduler = get_scheduler(
                                name="linear", 
                                optimizer=optimizer,
                                num_warmup_steps=0,        
                                num_training_steps=num_training_steps
                            )

    criterion = nn.CrossEntropyLoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # torch.set_num_threads(2)  
    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, filename='bbc_bert_benchmarking', device=device, benchmarking=True, scheduler=scheduler)

