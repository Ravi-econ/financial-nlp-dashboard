# ============================================================
# PHASE 3: SENTIMENT ANALYSIS
# File: phase3_sentiment.py
#
# What this file does:
#   - Runs TWO different sentiment analyzers on every article:
#       1. VADER  → Rule-based, fast, good baseline
#       2. FinBERT → Transformer-based, finance-specific, more accurate
#   - Compares their outputs side by side
#   - Saves results with sentiment labels and scores
#
# KEY CONCEPT — Why two models?
#   VADER scores text based on hand-crafted rules and a sentiment
#   dictionary. It's fast but doesn't understand context.
#   FinBERT is a version of BERT fine-tuned on financial text
#   (earnings calls, news, analyst reports). It understands that
#   "the stock fell sharply" is negative even without a dictionary.
#   Comparing both shows you model trade-offs — a great interview topic.
# ============================================================

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')   # Suppress HuggingFace progress warnings


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

INPUT_FILE  = "data/articles_clean.csv"     # Output from Phase 2
OUTPUT_FILE = "data/articles_sentiment.csv" # Output for Phase 4+

# FinBERT model from HuggingFace Model Hub
# This will download ~420MB the first time — it's cached after that
FINBERT_MODEL = "ProsusAI/finbert"


# ─────────────────────────────────────────
# MODEL 1: VADER Sentiment Analysis
# ─────────────────────────────────────────

def load_vader():
    """
    Loads the VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzer.

    VADER works by:
    1. Looking up each word in a sentiment dictionary (positive/negative scores)
    2. Applying rules for capitalization, punctuation, negation ("not good")
    3. Computing a compound score from -1.0 (most negative) to +1.0 (most positive)

    HOW TO INTERPRET compound score:
        >= 0.05  → Positive
        <= -0.05 → Negative
        Between  → Neutral
    """
    print("📦 Loading VADER sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer()
    print("✅ VADER ready!\n")
    return analyzer


def vader_sentiment(text: str, analyzer) -> dict:
    """
    Runs VADER on a single piece of text and returns structured results.

    Parameters:
        text     : The article text to analyze
        analyzer : VADER SentimentIntensityAnalyzer object

    Returns:
        Dictionary with keys: vader_score, vader_label
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'vader_score': 0.0, 'vader_label': 'neutral'}

    # polarity_scores returns: {'neg': 0.1, 'neu': 0.6, 'pos': 0.3, 'compound': 0.52}
    scores = analyzer.polarity_scores(text)

    compound = scores['compound']  # This is the overall score we care about

    # Map compound score to a human-readable label
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'vader_score': round(compound, 4),
        'vader_label': label
    }


def run_vader_on_dataframe(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    """
    Applies VADER sentiment analysis to every article in the DataFrame.
    Uses the original 'full_text' (not preprocessed) because VADER
    is rule-based and works better with natural sentence structure
    (capitalization, punctuation, etc. all matter to VADER).

    Parameters:
        df       : DataFrame with 'full_text' column
        analyzer : VADER analyzer object

    Returns:
        DataFrame with 'vader_score' and 'vader_label' columns added
    """
    print("🔄 Running VADER on all articles...")

    # Apply vader_sentiment to each row's full_text
    # pd.Series(results) unpacks the list of dicts into columns automatically
    results = df['full_text'].apply(lambda text: vader_sentiment(text, analyzer))
    vader_df = pd.DataFrame(results.tolist())

    # Add the new columns to our main DataFrame
    df['vader_score'] = vader_df['vader_score']
    df['vader_label'] = vader_df['vader_label']

    print(f"✅ VADER complete! Label distribution:")
    print(df['vader_label'].value_counts().to_string())
    print()

    return df


# ─────────────────────────────────────────
# MODEL 2: FinBERT Sentiment Analysis
# ─────────────────────────────────────────

def load_finbert():
    """
    Loads FinBERT from HuggingFace.

    FinBERT is BERT (Bidirectional Encoder Representations from Transformers)
    fine-tuned on ~10,000 financial news sentences labeled as:
    positive / negative / neutral

    HOW IT WORKS:
    - Text is broken into "tokens" (subwords)
    - Tokens are converted to vectors (numbers that encode meaning)
    - A neural network processes ALL tokens simultaneously (bidirectional)
    - Final classification layer outputs probabilities for 3 classes

    The HuggingFace 'pipeline' wraps all of this complexity into one function.
    """
    print("📦 Loading FinBERT (this downloads ~420MB on first run)...")
    finbert = pipeline(
        task="sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        max_length=512,         # BERT has a max token limit of 512
        truncation=True         # Truncate longer texts to 512 tokens
    )
    print("✅ FinBERT ready!\n")
    return finbert


def finbert_sentiment(text: str, finbert_pipeline) -> dict:
    """
    Runs FinBERT on a single piece of text.

    Parameters:
        text             : The article text to analyze
        finbert_pipeline : HuggingFace pipeline object

    Returns:
        Dictionary with keys: finbert_label, finbert_confidence
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'finbert_label': 'neutral', 'finbert_confidence': 0.0}

    # Truncate very long text manually before passing to model
    # (extra safety beyond the pipeline's truncation)
    text = text[:1000]

    try:
        # The pipeline returns: [{'label': 'positive', 'score': 0.94}]
        result = finbert_pipeline(text)[0]
        return {
            'finbert_label': result['label'].lower(),
            'finbert_confidence': round(result['score'], 4)
        }
    except Exception as e:
        # If the model fails on a specific text, return neutral
        print(f"  ⚠️  FinBERT error on one article: {e}")
        return {'finbert_label': 'neutral', 'finbert_confidence': 0.0}


def run_finbert_on_dataframe(df: pd.DataFrame, finbert_pipeline) -> pd.DataFrame:
    """
    Applies FinBERT to every article. Uses 'full_text' (not preprocessed)
    because BERT models expect natural sentence flow — they've learned
    from billions of words of normal text.

    Parameters:
        df               : DataFrame with 'full_text' column
        finbert_pipeline : FinBERT pipeline object

    Returns:
        DataFrame with 'finbert_label' and 'finbert_confidence' columns added
    """
    print("🔄 Running FinBERT on all articles (this takes a few minutes)...")

    total = len(df)
    results = []

    for i, text in enumerate(df['full_text']):
        # Show progress every 25 articles
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total}...")

        result = finbert_sentiment(text, finbert_pipeline)
        results.append(result)

    finbert_df = pd.DataFrame(results)
    df['finbert_label'] = finbert_df['finbert_label']
    df['finbert_confidence'] = finbert_df['finbert_confidence']

    print(f"\n✅ FinBERT complete! Label distribution:")
    print(df['finbert_label'].value_counts().to_string())
    print()

    return df


# ─────────────────────────────────────────
# STEP 3: Model Comparison & Agreement
# ─────────────────────────────────────────

def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column showing whether VADER and FinBERT AGREED on sentiment.

    Model agreement is a useful quality signal:
    - When both agree → high confidence in the prediction
    - When they disagree → article may be ambiguous or nuanced

    This comparison also shows you WHY transformers > rule-based
    models for specialized domains (finance).
    """
    # True if both models gave the same label
    df['models_agree'] = df['vader_label'] == df['finbert_label']

    agreement_rate = df['models_agree'].mean() * 100
    print(f"🤝 Model Agreement Rate: {agreement_rate:.1f}%")
    print("   (Where they disagree, FinBERT is usually more accurate for finance)\n")

    return df


# ─────────────────────────────────────────
# STEP 4: Print sample comparison
# ─────────────────────────────────────────

def print_comparison_examples(df: pd.DataFrame):
    """
    Prints examples where the two models DISAGREE.
    This is very insightful — shows why domain-specific models matter.
    """
    disagreements = df[df['models_agree'] == False].head(3)

    if len(disagreements) == 0:
        print("📊 Both models agreed on everything!")
        return

    print("─" * 70)
    print("🔍 DISAGREEMENT EXAMPLES (VADER vs FinBERT):")
    print("─" * 70)

    for i, row in disagreements.iterrows():
        print(f"\nArticle: {row['title'][:80]}...")
        print(f"  VADER   → {row['vader_label']} (score: {row['vader_score']})")
        print(f"  FinBERT → {row['finbert_label']} (confidence: {row['finbert_confidence']})")

    print("─" * 70)


# ─────────────────────────────────────────
# MAIN — Run everything in order
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🎭 PHASE 3: Sentiment Analysis")
    print("=" * 55)

    # Load preprocessed articles from Phase 2
    print(f"\n📂 Loading from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} articles\n")

    # ─── MODEL 1: VADER ───
    vader_analyzer = load_vader()
    df = run_vader_on_dataframe(df, vader_analyzer)

    # ─── MODEL 2: FinBERT ───
    finbert = load_finbert()
    df = run_finbert_on_dataframe(df, finbert)

    # ─── COMPARE ───
    df = compare_models(df)
    print_comparison_examples(df)

    # Save enriched DataFrame
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved sentiment results to '{OUTPUT_FILE}'")

    print("\n✅ Phase 3 Complete! Run phase4_ner.py next.")


if __name__ == "__main__":
    main()
