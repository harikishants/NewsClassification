import torch
from torch.utils.data import DataLoader
# import torch.nn.functional as F
from tqdm import tqdm
import json

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, path = 'checkpoints/bbc_bert.pt', start_epoch = 0, patience=3):

    model.to(device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0
    epoch_no_improve = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train() # train mode
        total_loss = 0
        correct = 0
        total = 0


        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad() # resetting gradient to zero for each batch
            outputs = model(input_ids, attention_mask) # calls forward() method
            loss = criterion(outputs, labels) # computes cross entropy loss
            loss.backward() # computes gradients of loss w.r.t all params
            optimizer.step() # updates params

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct/total
        val_acc = evaluate_model(model, val_loader, device)

        history["train_loss"].append(total_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[info] Epoch {epoch}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epoch_no_improve = 0
            save_model(model, optimizer, epoch, val_acc, path=path)
        else:
            epoch_no_improve += 1

        if epoch_no_improve > patience:
            print(f"[info] Early stopping since there is no improvement.")
            break

    with open("training_history_layers_4_batch_16_GELU.json", "w") as f:
        json.dump(history, f)

def evaluate_model(model, val_loader, device):
    model.eval() # eval mode: dropout, batchNorm is disabled
    correct = 0
    total = 0

    with torch.no_grad(): # disable gradient tracking
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1) # index of max logits for all datapoints in the batch
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct/total

def save_model(model, optimizer, epoch, accuracy, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'hyperparams': model.hparams,
    }
    torch.save(checkpoint, path)
    # print(f"[info] Model saved to {path}")


def load_model(model_class, optimizer_class, path='bert_model.pt', device='cpu'):
    checkpoint = torch.load(path, map_location=device)

    model = model_class(**checkpoint['hyperparams'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = optimizer_class(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"[info] Loaded model from {path}")
    return model, optimizer, checkpoint['epoch'], checkpoint['accuracy'], checkpoint['hyperparams']