import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

def classify_newsBM(text, model_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                        )

    input_ids =  torch.tensor(encoded['input_ids'], dtype = torch.long)
    attention_mask =  torch.tensor(encoded['attention_mask'], dtype=torch.long)

    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
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

