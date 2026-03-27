# ============================================================
# PHASE 4: NAMED ENTITY RECOGNITION (NER)
# File: phase4_ner.py
#
# What this file does:
#   - Uses spaCy to identify named entities in every article:
#       → Organizations (Apple, Federal Reserve, Goldman Sachs)
#       → People (Jerome Powell, Elon Musk)
#       → Locations (United States, Wall Street)
#       → Money ($500 billion, €2 trillion)
#       → Dates & Events
#   - Builds a frequency analysis: which entities appear most?
#   - Links entities to their sentiment (is Apple in good or bad news?)
#   - Saves a separate entity-level CSV for dashboard use
#
# WHY NER matters in finance:
#   Knowing WHICH companies appear in NEGATIVE news before the market
#   does is literally what quantitative hedge funds do.
# ============================================================

import pandas as pd
import spacy
from collections import Counter
import json


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

INPUT_FILE        = "data/articles_sentiment.csv"  # Output from Phase 3
OUTPUT_ARTICLES   = "data/articles_ner.csv"        # Articles with entity lists
OUTPUT_ENTITIES   = "data/entity_summary.csv"      # Entity frequency table

# Entity types we care about in financial news
# Full list: https://spacy.io/models/en#en_core_web_sm-labels
RELEVANT_ENTITY_TYPES = {
    'ORG',     # Organizations, companies, agencies (Apple, Fed, IMF)
    'PERSON',  # People (Jerome Powell, Janet Yellen)
    'GPE',     # Countries, cities, states (USA, China, New York)
    'MONEY',   # Monetary amounts ($4.5 billion)
    'PERCENT', # Percentage values (5.25%, up 12%)
    'EVENT',   # Named events (COVID-19, Great Recession)
    'LAW',     # Laws and regulations (Dodd-Frank, Basel III)
}


# ─────────────────────────────────────────
# STEP 1: Load spaCy model
# ─────────────────────────────────────────

def load_nlp():
    """
    Loads the spaCy English model for NER.

    spaCy's NER component is trained to recognize entities
    using a deep learning model (transition-based NER).
    It processes text left-to-right and uses surrounding
    context to identify entity boundaries and types.
    """
    print("📦 Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded!\n")
    return nlp


# ─────────────────────────────────────────
# STEP 2: Extract entities from one article
# ─────────────────────────────────────────

def extract_entities(text: str, nlp) -> list:
    """
    Runs spaCy NER on a single text and returns a list of
    (entity_text, entity_type) tuples.

    HOW spaCy NER WORKS:
    1. Text is tokenized into words
    2. Each token is tagged with POS (part of speech)
    3. The NER model uses IOB tagging:
       B-ORG = Beginning of an Organization entity
       I-ORG = Inside (continuation of) an Organization entity
       O     = Outside (not an entity)
    4. Consecutive B/I tags are merged into a single entity span

    Parameters:
        text : Raw article text (we use full_text, not preprocessed,
               because NER needs proper capitalization)
        nlp  : spaCy language model

    Returns:
        List of tuples like [("Apple", "ORG"), ("Jerome Powell", "PERSON")]
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    # Process the text — creates a Doc with entity annotations
    doc = nlp(text[:1000])  # Limit to 1000 chars for speed

    entities = []
    for ent in doc.ents:
        # ent.text  → the actual text of the entity ("Federal Reserve")
        # ent.label_ → the entity type ("ORG")

        # Only keep entity types we care about
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue

        # Clean the entity text
        entity_text = ent.text.strip()

        # Skip very short or numeric-only entities (noise)
        if len(entity_text) < 2:
            continue

        entities.append((entity_text, ent.label_))

    # Remove duplicates within the same article
    return list(set(entities))


# ─────────────────────────────────────────
# STEP 3: Run NER on all articles
# ─────────────────────────────────────────

def run_ner_on_dataframe(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Applies entity extraction to every article.

    Stores results as:
    - 'entities' : JSON string of [(text, type), ...] for the full list
    - 'org_entities': Just the organization names (most useful for finance)

    Parameters:
        df  : DataFrame with 'full_text' and 'finbert_label' columns
        nlp : spaCy model

    Returns:
        DataFrame with entity columns added
    """
    print("🔄 Extracting named entities from all articles...")

    total = len(df)
    all_entities = []
    org_entities_list = []

    for i, text in enumerate(df['full_text']):
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total}...")

        entities = extract_entities(text, nlp)
        all_entities.append(entities)

        # Extract just organization names for the finance-specific analysis
        orgs = [e[0] for e in entities if e[1] == 'ORG']
        org_entities_list.append(orgs)

    # Store entities as JSON strings (so they can be saved in CSV)
    df['entities'] = [json.dumps(e) for e in all_entities]
    df['org_entities'] = [json.dumps(o) for o in org_entities_list]

    print(f"\n✅ NER complete on {total} articles!\n")
    return df


# ─────────────────────────────────────────
# STEP 4: Build entity frequency table
# ─────────────────────────────────────────

def build_entity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary table of entity frequencies with their
    associated sentiment.

    For each unique entity, we track:
    - How many articles mention it
    - How many of those articles are positive / negative / neutral
    - Its "sentiment score" (proportion of negative mentions)

    This is the core of what a financial intelligence system does:
    "Company X appears in 47 articles, 82% of which are negative."

    Parameters:
        df : DataFrame with 'entities', 'finbert_label' columns

    Returns:
        Entity summary DataFrame sorted by frequency
    """
    print("📊 Building entity frequency and sentiment summary...")

    entity_records = []  # Will hold one record per (article, entity) pair

    for _, row in df.iterrows():
        # Parse the JSON string back into a list of tuples
        try:
            entities = json.loads(row['entities'])
        except:
            continue

        sentiment = row.get('finbert_label', 'neutral')

        for entity_text, entity_type in entities:
            entity_records.append({
                'entity': entity_text,
                'entity_type': entity_type,
                'sentiment': sentiment
            })

    # Convert to DataFrame
    entity_df = pd.DataFrame(entity_records)

    if entity_df.empty:
        print("⚠️  No entities found!")
        return pd.DataFrame()

    # ── Group by entity and compute statistics ──
    summary = entity_df.groupby(['entity', 'entity_type']).agg(
        mention_count=('sentiment', 'count'),                                 # Total mentions
        positive_count=('sentiment', lambda x: (x == 'positive').sum()),     # Positive mentions
        negative_count=('sentiment', lambda x: (x == 'negative').sum()),     # Negative mentions
        neutral_count=('sentiment', lambda x: (x == 'neutral').sum()),       # Neutral mentions
    ).reset_index()

    # ── Compute sentiment ratio (0 = all positive, 1 = all negative) ──
    summary['negative_ratio'] = (
        summary['negative_count'] / summary['mention_count']
    ).round(3)

    # ── Sort by mention count (most prominent entities first) ──
    summary = summary.sort_values('mention_count', ascending=False)
    summary = summary.reset_index(drop=True)

    print(f"✅ Found {len(summary)} unique entities\n")

    # Show top 10 most mentioned organizations
    top_orgs = summary[summary['entity_type'] == 'ORG'].head(10)
    print("🏢 Top 10 Most Mentioned Organizations:")
    print(top_orgs[['entity', 'mention_count', 'negative_ratio']].to_string(index=False))
    print()

    return summary


# ─────────────────────────────────────────
# STEP 5: Find entities in negative news
# ─────────────────────────────────────────

def flag_high_risk_entities(summary: pd.DataFrame, threshold: float = 0.6):
    """
    Identifies entities with a high proportion of negative news.
    These are potential "at-risk" companies or people in the news cycle.

    Parameters:
        summary   : Entity summary DataFrame
        threshold : Negative ratio above which we flag an entity (default 60%)
    """
    # Filter: only ORGs with 3+ mentions AND high negative ratio
    high_risk = summary[
        (summary['entity_type'] == 'ORG') &
        (summary['mention_count'] >= 3) &
        (summary['negative_ratio'] >= threshold)
    ]

    if len(high_risk) == 0:
        print(f"✅ No organizations with >{threshold*100:.0f}% negative news found.")
        return

    print(f"\n⚠️  HIGH-RISK ORGANIZATIONS (>{threshold*100:.0f}% negative mentions):")
    print("─" * 55)
    for _, row in high_risk.iterrows():
        print(f"  {row['entity']:<30} "
              f"Mentions: {row['mention_count']:>3}  "
              f"Negative: {row['negative_ratio']*100:.0f}%")


# ─────────────────────────────────────────
# MAIN — Run everything in order
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🏷️  PHASE 4: Named Entity Recognition (NER)")
    print("=" * 55)

    # Load from Phase 3
    print(f"\n📂 Loading from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} articles\n")

    # Load spaCy
    nlp = load_nlp()

    # Extract entities
    df = run_ner_on_dataframe(df, nlp)

    # Build entity frequency + sentiment summary
    entity_summary = build_entity_summary(df)

    # Flag high-risk entities
    flag_high_risk_entities(entity_summary, threshold=0.6)

    # Save both outputs
    df.to_csv(OUTPUT_ARTICLES, index=False)
    entity_summary.to_csv(OUTPUT_ENTITIES, index=False)

    print(f"\n💾 Saved:")
    print(f"   Articles with entities → '{OUTPUT_ARTICLES}'")
    print(f"   Entity summary         → '{OUTPUT_ENTITIES}'")

    print("\n✅ Phase 4 Complete! Run phase5_topic_modeling.py next.")


if __name__ == "__main__":
    main()
