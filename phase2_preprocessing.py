# ============================================================
# PHASE 2: TEXT PREPROCESSING
# File: phase2_preprocessing.py
#
# What this file does:
#   - Loads the raw articles CSV from Phase 1
#   - Cleans the text (remove URLs, special chars, numbers)
#   - Tokenizes text (splits sentences into words)
#   - Removes stop words (common words like "the", "is", "a")
#   - Lemmatizes tokens (reduces words to their root form)
#   - Saves a new CSV with a cleaned 'processed_text' column
#
# WHY preprocessing matters:
#   NLP models work on tokens (individual words/subwords).
#   Raw text has noise (HTML tags, punctuation, repeated spaces)
#   that hurts model performance. Cleaning = better results.
# ============================================================

import re
import pandas as pd
import spacy

# NLTK is used here for additional stop words
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

INPUT_FILE  = "data/articles.csv"       # Output from Phase 1
OUTPUT_FILE = "data/articles_clean.csv" # Output for Phase 3+

# ─────────────────────────────────────────
# STEP 1: Load spaCy model
# ─────────────────────────────────────────

def load_spacy_model():
    """
    Loads the spaCy English model.

    spaCy models are pre-trained on large English text.
    'en_core_web_sm' is the small (fast) version — good for our use case.
    It handles: tokenization, POS tagging, NER, lemmatization.

    If not installed, run: python -m spacy download en_core_web_sm
    """
    print("📦 Loading spaCy English model...")
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded!\n")
    return nlp


# ─────────────────────────────────────────
# STEP 2: Basic text cleaning (regex)
# ─────────────────────────────────────────

def basic_clean(text: str) -> str:
    """
    Applies basic regex-based cleaning to raw text.

    Regex (Regular Expressions) lets us find and replace patterns in text.
    We use it to strip out things that add no meaning for NLP.

    Parameters:
        text : Raw article text string

    Returns:
        Lightly cleaned string
    """
    if not isinstance(text, str):
        return ""  # Handle NaN / None values safely

    # Remove URLs (http://... or https://...)
    # Pattern: 'http' followed by non-space characters
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags like <p>, <b>, </div>
    text = re.sub(r'<.*?>', '', text)

    # Remove content inside square brackets like [+2456 chars] from NewsAPI
    text = re.sub(r'\[.*?\]', '', text)

    # Remove special characters — keep only letters, numbers, spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip().lower()

    return text


# ─────────────────────────────────────────
# STEP 3: Tokenize, remove stop words, lemmatize
# ─────────────────────────────────────────

# Load English stop words from NLTK
# Stop words = extremely common words that carry no meaning
# Examples: "the", "a", "is", "in", "to", "and"
STOP_WORDS = set(stopwords.words('english'))

# Add finance-specific stop words that are common but meaningless
CUSTOM_STOP_WORDS = {
    'said', 'say', 'year', 'new', 'also', 'percent',
    'would', 'could', 'one', 'two', 'three', 'first'
}
STOP_WORDS.update(CUSTOM_STOP_WORDS)


def advanced_preprocess(text: str, nlp) -> str:
    """
    Runs spaCy NLP pipeline on text to:
    1. Tokenize — break into individual words (tokens)
    2. Remove stop words — filter out meaningless common words
    3. Lemmatize — convert words to their dictionary root form

    LEMMATIZATION EXAMPLES:
        "running"  → "run"
        "prices"   → "price"
        "raised"   → "raise"
        "economies"→ "economy"

    Why lemmatize? So "inflation" and "inflationary" map to the same concept.

    Parameters:
        text : Already basic-cleaned string
        nlp  : spaCy language model object

    Returns:
        String of processed tokens joined by spaces
    """
    # spaCy processes the text and creates a Doc object
    # A Doc is a sequence of Token objects, each with many attributes
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip stop words
        if token.text in STOP_WORDS:
            continue

        # Skip punctuation (periods, commas, etc.)
        if token.is_punct:
            continue

        # Skip very short tokens (single letters add no value)
        if len(token.text) <= 2:
            continue

        # Keep the LEMMA (root form) of the token
        # token.lemma_ gives the base dictionary form
        tokens.append(token.lemma_)

    # Join all remaining tokens back into a single string
    return " ".join(tokens)


# ─────────────────────────────────────────
# STEP 4: Apply preprocessing to all articles
# ─────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to every article in the DataFrame.

    Pipeline per article:
        raw full_text → basic_clean() → advanced_preprocess() → processed_text

    Parameters:
        df  : DataFrame with 'full_text' column (from Phase 1)
        nlp : spaCy model

    Returns:
        DataFrame with new 'processed_text' column added
    """
    print("⚙️  Preprocessing all articles (this may take a minute)...")

    total = len(df)

    processed_texts = []

    for i, text in enumerate(df['full_text']):
        # Show progress every 50 articles
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total} articles processed...")

        # Step A: Basic cleaning (regex)
        cleaned = basic_clean(text)

        # Step B: Advanced NLP (tokenize + lemmatize + remove stop words)
        processed = advanced_preprocess(cleaned, nlp)

        processed_texts.append(processed)

    # Add the processed column to the DataFrame
    df['processed_text'] = processed_texts

    print(f"\n✅ All {total} articles preprocessed!\n")
    return df


# ─────────────────────────────────────────
# STEP 5: Show a before/after example
# ─────────────────────────────────────────

def show_example(df: pd.DataFrame):
    """
    Prints a before/after comparison so you can SEE what preprocessing did.
    This is very useful for understanding + for explaining in interviews.
    """
    print("─" * 60)
    print("📝 BEFORE vs AFTER Preprocessing Example:")
    print("─" * 60)

    sample = df.iloc[0]  # Take the first article as example

    print(f"\n🔴 BEFORE (raw full_text):\n{sample['full_text'][:300]}...")
    print(f"\n🟢 AFTER (processed_text):\n{sample['processed_text'][:300]}...")
    print("─" * 60)


# ─────────────────────────────────────────
# MAIN — Run everything in order
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🧹 PHASE 2: Text Preprocessing")
    print("=" * 55)

    # Load the CSV from Phase 1
    print(f"\n📂 Loading articles from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} articles\n")

    # Load spaCy NLP model
    nlp = load_spacy_model()

    # Run full preprocessing pipeline
    df = preprocess_dataframe(df, nlp)

    # Show a before/after comparison
    show_example(df)

    # Save the enriched DataFrame
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved preprocessed articles to '{OUTPUT_FILE}'")

    print("\n✅ Phase 2 Complete! Run phase3_sentiment.py next.")


if __name__ == "__main__":
    main()
