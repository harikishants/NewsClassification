import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

from dataset import BBCDataset
from tokenizer import BPETokenizer
from model import BERTClassifier
from train import train_model, load_model
from inference import classify_news

if __name__ == "__main__":

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


    tokenizer = BPETokenizer(vocab_file='vocab/vocab_final')

    train_dataset = BBCDataset(X_train, y_train, tokenizer)
    val_dataset = BBCDataset(X_val, y_val, tokenizer)

    # print(train_dataset.__getitem__(2))
    # exit()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # model = BERTClassifier(vocab_size = len(tokenizer.vocab), num_layers=4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, filename='bbc_bert_layers_4_batch_16_GELU', device=device)

    # re-training
    model, optimizer, epoch, accuracy, hyperparams = load_model(BERTClassifier, torch.optim.AdamW, filename='bbc_bert_0.8449_layer_2', device='cuda')

    # summary(model, input_size=(16, 512), device='cuda')
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 2e-5
    # train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, filename = 'bbc_bert_layer_4_lr_2e-5', device=device, start_epoch=epoch+1)
