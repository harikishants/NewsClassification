import warnings
warnings.filterwarnings('ignore')

import torch

from tokenizer import BPETokenizer
from model import BERTClassifier
from train import train_model, load_model

def classify_news(sentence, model_path):

    tokenizer = BPETokenizer(vocab_file='vocab/vocab_final')
    model, optimizer, epoch, accuracy, hyperparams = load_model(BERTClassifier, torch.optim.AdamW, file_path=model_path, device='cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    max_len = hyperparams['max_len']

    tokenized_output = tokenizer.tokenize(sentence) # gives splitted text and its IDs
    token_ids = tokenized_output.ids
    token_ids = token_ids[ : max_len] # truncating to max length
    attention_mask = [1] * len(token_ids) # bert: attention on both sides

    # padding
    pad_length = max_len - len(token_ids)
    pad_id = tokenizer.get_pad_id()
    token_ids += [pad_id] * pad_length # padding right side
    attention_mask += [0] * pad_length


    inputs_ids = torch.tensor([token_ids], dtype = torch.long).to(device) # single datapoint, not a batch
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(inputs_ids, attention_mask)
        # print(outputs)
        predicted_label = outputs.argmax(dim=1).item()

    label_map = {
                'business': 0,
                'entertainment': 1,
                'politics': 2,
                'sport': 3,
                'tech': 4
                }
    
    reversed_label_map = {value: key for key, value in label_map.items()}
    predicted_class = reversed_label_map.get(predicted_label)
    
    return predicted_class

