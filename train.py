import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}")
        save_model(model, optimizer, epoch, val_acc, path=f'checkpoints/bbc_bert_epoch_{epoch}.pt')

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct/total

def save_model(model, optimizer, epoch, accuracy, path='bert_model.pt'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'hyperparams': model.hparams,
    }
    torch.save(checkpoint, path)
    print(f"[info] Model saved to {path}")


# def load_model(model_class, optimizer_class, path='bert_model.pt', device='cpu'):
#     checkpoint = torch.load(path, map_location=device)

#     model = model_class(**checkpoint['hyperparams'])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)

#     optimizer = optimizer_class(model.parameters())
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     print(f"[info] Loaded model from {path}")
#     return model, optimizer, checkpoint['epoch'], checkpoint['accuracy'], checkpoint['hyperparams']