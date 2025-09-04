# NLP Sentiment & Fine-Tuning Projects

This repo contains two distinct workflows for sentiment classification:
1. A traditional machine learning pipeline using TF-IDF + Logistic Regression.
2. A transformer-based fine-tuning pipeline using Hugging Face + BERT.

Both are applied to real sentiment data and include full preprocessing, training, evaluation and insight generation.

---

## Project Goals

By examining both pipelines, the repo helps understand the trade-offs between model complexity and interpretability, practical steps for real-world NLP and how transformer fine-tuning differs from classical vectorization.

---

## Notebook Summaries

### Sentiment_Analysis.ipynb *(Classical ML)*  
**Goal**: Build an interpretable sentiment classifier from scratch using scikit-learn.

#### 📘 Dataset:
- Assumes a labeled dataset of short text reviews (e.g., from IMDb or similar)
- Columns typically: `text`, `label` (binary: positive/negative)

#### 🧹 Preprocessing:
- Text is cleaned, tokenized, and lowercased.
- Stop words are removed.
- Data is split into training and test sets.

#### 🧠 Modeling:
- TF-IDF vectorization transforms text into sparse numerical features.
- A logistic regression classifier is trained to distinguish sentiment.
- Includes a grid search for hyperparameter tuning (`C` values).

#### 📊 Evaluation:
- Performance is measured using accuracy, precision, recall, and F1-score.
- Confusion matrix gives insight into types of misclassifications.

#### 📌 Insights:
- Easy to build and deploy.
- Works well on smaller datasets.
- Limited in capturing semantic nuances like sarcasm or negation.

---

### 2️⃣ finetunning.ipynb — *(Transformer Fine-Tuning)*  
**Goal**: Fine-tune a pretrained BERT model for sentiment classification using Hugging Face Transformers.

#### 📘 Dataset:
- Same format: `text`, `label`
- Typically larger than classical ML notebooks to benefit from deep learning

#### 🧹 Preprocessing:
- Uses `AutoTokenizer` to tokenize input into WordPiece format.
- Labels are converted to integers.

#### ⚙️ Model & Training:
- Loads `bert-base-uncased` via `AutoModelForSequenceClassification`
- Uses `Trainer` API for training loop
- Includes training arguments: learning rate, batch size, epochs
- GPU-compatible and Colab-ready

#### 📊 Evaluation:
- Metrics include accuracy, precision, recall, F1-score (same as classical)
- Training/validation loss curves and logs
- Option to export the trained model

#### 📌 Insights:
- Stronger performance with limited feature engineering
- Learns contextual meaning of words (e.g., “not good” ≠ “good”)
- Requires more compute and memory
- Ideal for production applications with nuance in language

---

## 📈 Results Summary

| Method                  | Accuracy | F1 Score | Notes                          |
|------------------------|----------|----------|--------------------------------|
| TF-IDF + Logistic Reg. | ~82%     | ~0.84    | Fast, interpretable baseline   |
| Fine-Tuned BERT        | ~89%     | ~0.90    | Stronger generalization        |

> 🎯 BERT outperforms classic methods on complex sentiment — especially when context matters.

---

## ⚙️ Setup

```bash
git clone https://github.com/CassandraMaldonado/SentimentTune.git
cd SentimentTune

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
