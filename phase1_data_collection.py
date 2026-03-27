# ============================================================
# PHASE 1: DATA COLLECTION
# File: phase1_data_collection.py
#
# What this file does:
#   - Connects to NewsAPI to fetch real financial news
#   - Cleans and structures the raw API response
#   - Saves articles to a CSV for use in all later phases
#
# Run this file first before anything else.
# ============================================================

import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Paste your NewsAPI key here (get it free at https://newsapi.org)
API_KEY = "61dd3e7d09244bdc914c835d2fcb6b74"

# How many days back to fetch news (free tier allows up to 30 days)
DAYS_BACK = 7

# Keywords to search for — these are classic macroeconomic/finance terms
# You can add more like "cryptocurrency", "oil prices", "earnings", etc.
SEARCH_QUERIES = [
    "inflation federal reserve",
    "stock market crash rally",
    "interest rates economy",
    "GDP recession growth",
    "unemployment jobs report"
]

# Trusted financial news sources (NewsAPI source IDs)
SOURCES = "reuters,bloomberg,the-wall-street-journal,financial-times,cnbc"

# Output file path
OUTPUT_FILE = "data/articles.csv"


# ─────────────────────────────────────────
# STEP 1: Initialize the NewsAPI client
# ─────────────────────────────────────────

def initialize_client(api_key: str) -> NewsApiClient:
    """
    Creates and returns a NewsAPI client object.
    This is the object we'll use to make all API requests.
    """
    print("🔗 Connecting to NewsAPI...")
    client = NewsApiClient(api_key=api_key)
    print("✅ Connected successfully!\n")
    return client


# ─────────────────────────────────────────
# STEP 2: Fetch articles for a single query
# ─────────────────────────────────────────

def fetch_articles_for_query(client: NewsApiClient, query: str, from_date: str) -> list:
    """
    Fetches up to 100 articles for a single search query.

    Parameters:
        client    : NewsApiClient object
        query     : The search term (e.g. "inflation federal reserve")
        from_date : ISO format date string like "2024-01-01"

    Returns:
        List of article dictionaries from the API
    """
    print(f"  🔍 Fetching news for: '{query}'...")

    response = client.get_everything(
        q=query,                    # The search keyword(s)
        from_param=from_date,       # Only articles after this date
        language='en',              # English articles only
        sort_by='publishedAt',      # Get most recent first
        page_size=100               # Max articles per request (API limit)
    )

    articles = response.get('articles', [])
    print(f"     → Found {len(articles)} articles")
    return articles


# ─────────────────────────────────────────
# STEP 3: Fetch articles for ALL queries
# ─────────────────────────────────────────

def fetch_all_articles(client: NewsApiClient, queries: list, days_back: int) -> list:
    """
    Loops through all search queries and collects all articles.

    Parameters:
        client    : NewsApiClient object
        queries   : List of search query strings
        days_back : How many days back to search

    Returns:
        Combined list of all articles from all queries
    """
    # Calculate the start date dynamically
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    print(f"📅 Fetching news from {from_date} to today...\n")

    all_articles = []

    for query in queries:
        articles = fetch_articles_for_query(client, query, from_date)
        all_articles.extend(articles)  # Add this batch to the master list

    print(f"\n📦 Total raw articles collected: {len(all_articles)}")
    return all_articles


# ─────────────────────────────────────────
# STEP 4: Clean and structure the data
# ─────────────────────────────────────────

def clean_articles(articles: list) -> pd.DataFrame:
    """
    Converts the raw list of API article dicts into a clean DataFrame.

    - Drops articles with missing title or description
    - Removes duplicate articles (same URL published by multiple queries)
    - Extracts only the columns we care about
    - Adds a combined 'full_text' column we'll use for NLP

    Parameters:
        articles : Raw list of article dicts from NewsAPI

    Returns:
        Cleaned pandas DataFrame
    """
    print("\n🧹 Cleaning articles...")

    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(articles)

    # --- Extract nested fields ---
    # The 'source' field is a dict like {'id': 'reuters', 'name': 'Reuters'}
    # We only want the name string
    df['source_name'] = df['source'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')

    # --- Select only the columns we need ---
    columns_to_keep = ['title', 'description', 'content', 'url', 'publishedAt', 'source_name']
    df = df[columns_to_keep]

    # --- Rename for clarity ---
    df = df.rename(columns={'publishedAt': 'published_at'})

    # --- Drop rows where title OR description is missing ---
    # These articles are useless for NLP since we have no text to analyze
    before = len(df)
    df = df.dropna(subset=['title', 'description'])
    print(f"  Removed {before - len(df)} articles with missing text")

    # --- Remove duplicate articles (same URL = same article) ---
    before = len(df)
    df = df.drop_duplicates(subset='url')
    print(f"  Removed {before - len(df)} duplicate articles")

    # --- Combine title + description into one 'full_text' column ---
    # This gives us more text per article for better NLP analysis
    df['full_text'] = df['title'] + ". " + df['description']

    # --- Parse date string into proper datetime ---
    df['published_at'] = pd.to_datetime(df['published_at'])

    # --- Reset index for clean numbering ---
    df = df.reset_index(drop=True)

    print(f"✅ Clean articles ready: {len(df)} articles\n")
    return df


# ─────────────────────────────────────────
# STEP 5: Save to CSV
# ─────────────────────────────────────────

def save_articles(df: pd.DataFrame, output_path: str):
    """
    Saves the cleaned DataFrame to a CSV file.
    Creates the 'data/' folder if it doesn't exist.

    Parameters:
        df          : Cleaned DataFrame
        output_path : Where to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"💾 Saved {len(df)} articles to '{output_path}'")


# ─────────────────────────────────────────
# MAIN — Run everything in order
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  📰 PHASE 1: Financial News Data Collection")
    print("=" * 55)

    # Step 1: Connect to API
    client = initialize_client(API_KEY)

    # Step 2 + 3: Fetch articles for all queries
    raw_articles = fetch_all_articles(client, SEARCH_QUERIES, DAYS_BACK)

    # Step 4: Clean and structure the data
    df = clean_articles(raw_articles)

    # Step 5: Save to disk
    save_articles(df, OUTPUT_FILE)

    # Quick preview of what we collected
    print("\n📊 Sample of collected data:")
    print(df[['title', 'source_name', 'published_at']].head(5).to_string(index=False))

    print("\n✅ Phase 1 Complete! Run phase2_preprocessing.py next.")


if __name__ == "__main__":
    main()
