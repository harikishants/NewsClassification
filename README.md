# 📰 BBC News Classification using Custom and Pretrained BERT (PyTorch)

This project implements a complete NLP pipeline for classifying BBC News articles into five categories: `business`, `entertainment`, `politics`, `sport`, and `tech`. It demonstrates both a custom-built BERT model from scratch using PyTorch and a benchmark using Hugging Face’s pre-trained BERT.

---

## 🔧 Project Structure

```
├── data/
│   └── bbc-news-data.csv       # CSV dataset from Kaggle containing BBC news articles
├── checkpoints/                # Saved model checkpoints
├── dataset.py                  # Custom PyTorch Dataset classes (BBCDataset for custom tokenization, BBCDatasetBM for Hugging Face tokenization)
├── tokenizer.py                # Custom Byte Pair Encoding (BPE) Tokenizer (BPETokenizer)
├── model.py                    # Custom BERT classifier implemented from scratch (BERTClassifier)
├── train.py                    # Training loop and utilities (model saving & loading)
├── main.py                     # Main script for training/evaluation with the custom BERT model
├── benchmarking.py             # Benchmark training using Hugging Face's BertForSequenceClassification
├── inference.py                # Inference functions for custom BERT
├── inferenceBM.py              # Inference functions using Hugging Face BERT
├── test_tokenizer.py           # Tests for the BPETokenizer
├── test_inference.py           # Tests for the inference pipelines
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🗃 Dataset

The BBC News Classification dataset contains 2225 articles, each labeled into one of the five categories. The dataset CSV file is placed in the `data/` directory and includes the following columns:
- `category`
- `content`

---

## 🧠 Models

### 1. Custom BERT Model
- **File:** `model.py`
- **Description:** Implements a BERT classifier (`BERTClassifier`) from scratch using token and positional embeddings, multi-head self-attention, Transformer encoder layers, and a classification head.

### 2. Hugging Face BERT Benchmark
- **Files:** `benchmarking.py` and `inferenceBM.py`
- **Description:** Uses the pre-trained `bert-base-uncased` model with a sequence classification head (`BertForSequenceClassification`), fine-tuned on the BBC News dataset.

---

## 🔤 Tokenization

- **File:** `tokenizer.py`
- **Description:** Implements the BPETokenizer which:
  - Builds a vocabulary from the input corpus using Byte Pair Encoding (BPE),
  - Supports merging of adjacent tokens,
  - Saves and loads the vocabulary to/from JSON,
  - Tokenizes input text to generate token IDs for training and inference.

---

## 🏁 Training & Evaluation

- **Training Custom BERT:**  
  The training loop, along with checkpoint saving, early stopping, and evaluation, is implemented in `train.py` and integrated via `main.py`.

- **Benchmarking with Hugging Face BERT:**  
  Training and evaluation for the Hugging Face BERT model are managed in `benchmarking.py`.

---

## 🔎 Inference

- **Custom BERT Inference:**  
  See `inference.py` for using the custom BERT model and BPETokenizer to classify new articles.

- **Hugging Face BERT Inference:**  
  See `inferenceBM.py` for using the Hugging Face pipeline to infer news classification.

---

## 🚀 Running the Project

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training Custom BERT**
   ```bash
   python main.py
   ```

3. **Benchmarking with Hugging Face BERT**
   ```bash
   python benchmarking.py
   ```

4. **Custom Inference Example**
   ```python
   from inference import classify_news
   predicted_class = classify_news("The government announced new policy changes.", model_path='checkpoints/your_custom_bert_model.pt')
   print(predicted_class)
   ```

5. **Hugging Face Inference Example**
   ```python
   from inferenceBM import classify_newsBM
   predicted_class = classify_newsBM("Apple released a new MacBook with advanced chips.", model_path='checkpoints/your_hf_bert_model.pt')
   print(predicted_class)
   ```

---

## 📜 License

MIT License – Free to use for educational and research purposes.

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BBC News Dataset from Kaggle](https://www.kaggle.com/datasets/cpichotta/bbc-news-data)