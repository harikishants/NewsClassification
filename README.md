# 📰 BBC News Classification using Custom and Pretrained BERT (PyTorch)

This project demonstrates a complete NLP pipeline for classifying BBC News articles into five categories: `business`, `entertainment`, `politics`, `sport`, and `tech`. It includes:

- ✅ A **BERT model built from scratch** (token embeddings, positional encoding, encoder layers, classifier head)
- 🚀 A **benchmark** using the Hugging Face `BertForSequenceClassification` model
- 📊 Training, evaluation, and inference pipelines
- 🗂 Custom dataset class using PyTorch `Dataset` and `DataLoader`

---

## 🔧 Project Structure

```
├── dataset.py                 # Custom PyTorch Dataset for BBC News
├── bpe_tokenizer.py          # Custom Byte Pair Encoding tokenizer (if applicable)
├── custom_bert.py            # BERT model implemented from scratch
├── train_custom_bert.py      # Training loop for custom BERT
├── train_hf_bert.py          # Training loop for Hugging Face BERT
├── inference_custom.py       # Inference function for custom BERT
├── inference_hf.py           # Inference function for Hugging Face BERT
├── utils.py                  # Helper utilities (saving, loading, evaluation)
├── checkpoints/              # Saved model checkpoints
├── data/
│   └── bbc-news-data.csv     # Input dataset (5-class BBC News)
├── requirements.txt
└── README.md
```

---

## 🗃 Dataset

The dataset is from the [BBC News Classification Dataset](https://www.kaggle.com/datasets/cpichotta/bbc-news-data). It contains 2225 articles labeled into 5 categories.

- **Format**: `.csv` file with columns: `category`, `content`

---

## 🧠 Models

### 1. Custom BERT
Implemented from scratch using PyTorch:
- Token + Positional Embeddings
- Multi-head Self Attention
- Feedforward layers
- Encoder blocks
- Classification head

### 2. Hugging Face BERT
- `bert-base-uncased` with a classifier head (`BertForSequenceClassification`)
- Fine-tuned on BBC News dataset with:
  - Classifier-specific learning rate
  - Warmup + LR scheduler
  - Evaluation using validation accuracy

---

## 🏁 Training

### Train Custom BERT
```bash
python train_custom_bert.py
```

### Train Hugging Face BERT
```bash
python train_hf_bert.py
```

Both scripts support:
- Early stopping
- Checkpoint saving
- Training history logging

---

## 🔎 Inference

### Custom BERT Inference
```python
from inference_custom import classify_news_custom
classify_news_custom("The government announced new policy changes.")
```

### Hugging Face Inference
```python
from inference_hf import classify_news_hf
classify_news_hf("Apple released a new MacBook with advanced chips.")
```

---

## 📊 Results

| Model             | Validation Accuracy |
|-------------------|-------------------|
| Custom BERT (4L)  | **~88%**         |
| Hugging Face BERT | **~95%**         |

**Note**: Hugging Face BERT took longer to converge and required LR tuning & scheduler.

---

## 💡 Key Learnings

- Building BERT from scratch deepens understanding of transformer internals
- Fine-tuning pretrained models can outperform scratch models with fewer epochs
- Learning rate scheduling & per-layer tuning can greatly affect performance
- Custom models can be competitive with good architecture & training

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📜 License

MIT License. Free to use for educational and research purposes.

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BBC News Dataset from Kaggle](https://www.kaggle.com/datasets/cpichotta/bbc-news-data)
