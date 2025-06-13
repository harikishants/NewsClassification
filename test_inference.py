
from inference import classify_news
from inferenceBM import classify_newsBM

if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        sentence = f.read().strip()

    predicted_class = classify_news(sentence, model_path = 'checkpoints/bbc_bert_0.8449_layer_2.pt')
    predicted_classBM = classify_newsBM(sentence, model_path = 'checkpoints/bbc_bert_benchmarking.pt')
    
    print(f'Predicted class = {predicted_class}')
    print(f'Predicted classBM = {predicted_classBM}')