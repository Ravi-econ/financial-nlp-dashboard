# 📈 Financial News Sentiment Analyzer

> An end-to-end NLP pipeline that collects real financial news, runs sentiment analysis, 
> extracts named entities, discovers hidden topics, and visualizes everything in a live dashboard.

---

## 🧠 What This Project Covers

| NLP Concept | Tool Used | File |
|---|---|---|
| Text Preprocessing | spaCy, NLTK | `phase2_preprocessing.py` |
| Sentiment Analysis (Rule-based) | VADER | `phase3_sentiment.py` |
| Sentiment Analysis (Transformer) | FinBERT | `phase3_sentiment.py` |
| Named Entity Recognition | spaCy NER | `phase4_ner.py` |
| Topic Modeling | LDA (scikit-learn) | `phase5_topic_modeling.py` |
| TF-IDF Vectorization | scikit-learn | `phase5_topic_modeling.py` |
| Dashboard & Visualization | Streamlit + Plotly | `dashboard.py` |

---

## 🚀 Setup & Installation

### 1. Clone the project
```bash
git clone https://github.com/yourusername/financial-nlp-dashboard
cd financial-nlp-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add your NewsAPI key
Open `phase1_data_collection.py` and replace:
```python
API_KEY = "YOUR_API_KEY_HERE"
```
Get a free key at [https://newsapi.org](https://newsapi.org)

---

## ▶️ Running the Pipeline

Run each phase in order:

```bash
# Phase 1: Collect news articles
python phase1_data_collection.py

# Phase 2: Clean and preprocess text
python phase2_preprocessing.py

# Phase 3: Run sentiment analysis (VADER + FinBERT)
python phase3_sentiment.py

# Phase 4: Extract named entities
python phase4_ner.py

# Phase 5: Topic modeling (LDA)
python phase5_topic_modeling.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
financial-nlp-dashboard/
│
├── phase1_data_collection.py   # NewsAPI → raw articles CSV
├── phase2_preprocessing.py     # Raw text → clean tokens
├── phase3_sentiment.py         # VADER + FinBERT sentiment scores
├── phase4_ner.py               # spaCy named entity extraction
├── phase5_topic_modeling.py    # LDA topic discovery + word clouds
├── dashboard.py                # Streamlit interactive dashboard
├── requirements.txt
│
├── data/                       # Auto-created by pipeline
│   ├── articles.csv
│   ├── articles_clean.csv
│   ├── articles_sentiment.csv
│   ├── articles_ner.csv
│   ├── articles_topics.csv     ← Final dataset
│   └── entity_summary.csv
│
└── visualizations/             # Auto-created word cloud PNGs
```

---

## 💡 Key NLP Concepts Demonstrated

- **Tokenization**: Splitting raw text into individual words
- **Stop Word Removal**: Filtering high-frequency, low-meaning words
- **Lemmatization**: Reducing words to their dictionary root form
- **TF-IDF**: Measuring word importance relative to a corpus
- **Transformer Models**: Using pre-trained BERT for context-aware NLP
- **IOB Tagging**: How spaCy identifies entity boundaries
- **LDA Topic Modeling**: Unsupervised discovery of latent topics
- **Perplexity**: Evaluating topic model quality

---

## 🌐 Deploying the Dashboard

1. Push your project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Deploy — you'll get a public URL like `https://yourname-financial-nlp.streamlit.app`

---

## 🤝 Interview Talking Points

- "I compared rule-based (VADER) and transformer-based (FinBERT) sentiment models"
- "FinBERT outperforms VADER on financial text because it was fine-tuned on earnings calls and financial news"
- "I used LDA topic modeling to discover latent economic themes without any labeled data"
- "The NER pipeline flags companies with high negative news exposure — analogous to what quant funds do"
- "I deployed the full pipeline as an interactive Streamlit dashboard"
