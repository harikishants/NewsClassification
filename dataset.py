import torch
from torch.utils.data import Dataset

class BBCDataset(Dataset): # for custom built tokenizer
    def __init__(self, texts, labels, tokenizer, max_len = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokenized_output = self.tokenizer.tokenize(text) # gives splitted text and its IDs
        token_ids = tokenized_output.ids
        token_ids = token_ids[ : self.max_len] # truncating to max length
        attention_mask = [1] * len(token_ids) # bert: attention on both sides

        # padding
        pad_length = self.max_len - len(token_ids)
        pad_id = self.tokenizer.get_pad_id()
        token_ids += [pad_id] * pad_length # padding right side
        attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(token_ids, dtype = torch.long), # single datapoint, not a batch
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype = torch.long)
        }
    
class BBCDatasetBM(Dataset):# for benchmarking using pre-trained bert tokenizer
    def __init__(self, texts, labels, tokenizer, max_len = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
                                text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len,
                                return_tensors='pt'
                                )

        return {
            "input_ids": torch.tensor(encoded['input_ids'].squeeze(), dtype = torch.long), # single datapoint, not a batch
            "attention_mask": torch.tensor(encoded['attention_mask'].squeeze(), dtype=torch.long),
            "label": torch.tensor(label, dtype = torch.long)
        }
