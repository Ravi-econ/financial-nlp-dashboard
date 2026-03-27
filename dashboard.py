# ============================================================
# STREAMLIT DASHBOARD
# File: dashboard.py
#
# What this file does:
#   - Builds an interactive web dashboard for all NLP results
#   - Shows real-time sentiment overview (bull/bear market indicator)
#   - Visualizes sentiment trends over time
#   - Displays entity analysis (which companies are in the news)
#   - Shows topic clusters discovered by LDA
#   - Lets you search and filter individual articles
#
# How to run:
#   streamlit run dashboard.py
#
# This opens a browser at http://localhost:8501
# Deploy publicly for FREE at https://streamlit.io/cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime


# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Financial NLP Dashboard",
    page_icon="📈",
    layout="wide",              # Use full browser width
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 5px 0;
    }
    .positive { border-left-color: #00C851; }
    .negative { border-left-color: #FF4444; }
    .neutral  { border-left-color: #FFBB33; }
    .big-font { font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING (cached for performance)
# ─────────────────────────────────────────

@st.cache_data  # Cache decorator: Streamlit only re-runs this when data changes
def load_data():
    """
    Loads all CSV outputs from the NLP pipeline phases.

    @st.cache_data means the data is loaded ONCE and reused across
    every user interaction — makes the dashboard fast and responsive.

    Returns:
        Tuple of (articles_df, entity_df) or (None, None) if files missing
    """
    articles_path = "data/articles_topics.csv"
    entities_path = "data/entity_summary.csv"

    if not os.path.exists(articles_path):
        return None, None

    articles_df = pd.read_csv(articles_path)

    # Parse the published_at column as datetime
    articles_df['published_at'] = pd.to_datetime(articles_df['published_at'], errors='coerce')

    # Parse date for grouping
    articles_df['date'] = articles_df['published_at'].dt.date

    entity_df = None
    if os.path.exists(entities_path):
        entity_df = pd.read_csv(entities_path)

    return articles_df, entity_df


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_market_signal(df: pd.DataFrame) -> tuple:
    """
    Computes the overall market sentiment signal from all articles.

    Returns:
        (signal_label, signal_emoji, positive_pct, negative_pct, neutral_pct)
    """
    counts = df['finbert_label'].value_counts(normalize=True) * 100

    pos = counts.get('positive', 0)
    neg = counts.get('negative', 0)
    neu = counts.get('neutral', 0)

    # Simple rule: whichever sentiment dominates sets the signal
    if pos > neg and pos > neu:
        return "BULLISH", "🟢", pos, neg, neu
    elif neg > pos and neg > neu:
        return "BEARISH", "🔴", pos, neg, neu
    else:
        return "NEUTRAL", "🟡", pos, neg, neu


def color_sentiment(label: str) -> str:
    """Maps sentiment label to an HTML color code."""
    colors = {'positive': '#00C851', 'negative': '#FF4444', 'neutral': '#FFBB33'}
    return colors.get(label, '#AAAAAA')


# ─────────────────────────────────────────
# SECTION 1: HEADER + KEY METRICS
# ─────────────────────────────────────────

def render_header(df: pd.DataFrame):
    """
    Renders the top section: title, market signal, and key metrics.
    """
    st.title("📈 Financial News Sentiment Dashboard")
    st.caption(f"Powered by FinBERT + spaCy NLP Pipeline | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    # Overall market signal
    signal, emoji, pos, neg, neu = get_market_signal(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📡 Market Signal",
            value=f"{emoji} {signal}",
            help="Determined by majority sentiment across all analyzed articles"
        )

    with col2:
        st.metric(
            label="📰 Articles Analyzed",
            value=f"{len(df):,}",
            help="Total articles collected and processed through the NLP pipeline"
        )

    with col3:
        sources = df['source_name'].nunique() if 'source_name' in df.columns else "N/A"
        st.metric(
            label="🗞️ News Sources",
            value=sources,
            help="Number of unique news outlets"
        )

    with col4:
        avg_confidence = df['finbert_confidence'].mean() if 'finbert_confidence' in df.columns else 0
        st.metric(
            label="🎯 Avg. FinBERT Confidence",
            value=f"{avg_confidence:.1%}",
            help="Average confidence score of FinBERT's sentiment predictions"
        )

    st.divider()


# ─────────────────────────────────────────
# SECTION 2: SENTIMENT BREAKDOWN
# ─────────────────────────────────────────

def render_sentiment_section(df: pd.DataFrame):
    """
    Renders sentiment distribution charts:
    - Overall pie/donut chart
    - Sentiment over time (line chart)
    - VADER vs FinBERT comparison
    """
    st.header("🎭 Sentiment Analysis")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # ── Donut chart: overall sentiment breakdown ──
        sentiment_counts = df['finbert_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_donut = px.pie(
            sentiment_counts,
            names='Sentiment',
            values='Count',
            hole=0.5,                           # Makes it a donut chart
            color='Sentiment',
            color_discrete_map={
                'positive': '#00C851',
                'negative': '#FF4444',
                'neutral':  '#FFBB33'
            },
            title="Overall Sentiment (FinBERT)"
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        # ── Line chart: sentiment over time ──
        if 'date' in df.columns:
            # Group by date and sentiment to get daily counts
            time_df = df.groupby(['date', 'finbert_label']).size().reset_index(name='count')

            fig_time = px.line(
                time_df,
                x='date',
                y='count',
                color='finbert_label',
                color_discrete_map={
                    'positive': '#00C851',
                    'negative': '#FF4444',
                    'neutral':  '#FFBB33'
                },
                title="Sentiment Trend Over Time",
                labels={'count': 'Article Count', 'date': 'Date', 'finbert_label': 'Sentiment'}
            )
            fig_time.update_layout(height=350, legend_title="Sentiment")
            st.plotly_chart(fig_time, use_container_width=True)

    # ── VADER vs FinBERT comparison bar chart ──
    if 'vader_label' in df.columns and 'finbert_label' in df.columns:
        st.subheader("🔬 Model Comparison: VADER vs FinBERT")

        vader_counts = df['vader_label'].value_counts(normalize=True) * 100
        finbert_counts = df['finbert_label'].value_counts(normalize=True) * 100

        labels = ['positive', 'negative', 'neutral']

        fig_compare = go.Figure(data=[
            go.Bar(
                name='VADER (Rule-Based)',
                x=labels,
                y=[vader_counts.get(l, 0) for l in labels],
                marker_color=['#00C851', '#FF4444', '#FFBB33'],
                opacity=0.6
            ),
            go.Bar(
                name='FinBERT (Transformer)',
                x=labels,
                y=[finbert_counts.get(l, 0) for l in labels],
                marker_color=['#00C851', '#FF4444', '#FFBB33'],
                opacity=1.0
            )
        ])

        fig_compare.update_layout(
            barmode='group',
            title="Sentiment Distribution: VADER vs FinBERT (%)",
            yaxis_title="Percentage of Articles (%)",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Agreement rate
        agreement = (df['vader_label'] == df['finbert_label']).mean() * 100
        st.info(f"📊 **Model Agreement Rate: {agreement:.1f}%** — Where they disagree, FinBERT is typically more accurate for financial text.")


# ─────────────────────────────────────────
# SECTION 3: ENTITY ANALYSIS
# ─────────────────────────────────────────

def render_entity_section(entity_df: pd.DataFrame):
    """
    Renders the Named Entity Recognition (NER) analysis section:
    - Top organizations in the news
    - Sentiment breakdown per organization
    - High-risk entities (high negative ratio)
    """
    st.header("🏷️ Named Entity Analysis")

    if entity_df is None or entity_df.empty:
        st.warning("No entity data found. Run phase4_ner.py first.")
        return

    # Filter to organizations only
    orgs = entity_df[entity_df['entity_type'] == 'ORG'].head(20)
    people = entity_df[entity_df['entity_type'] == 'PERSON'].head(10)

    col_left, col_right = st.columns(2)

    with col_left:
        # ── Horizontal bar chart: top organizations by mention count ──
        fig_orgs = px.bar(
            orgs,
            x='mention_count',
            y='entity',
            orientation='h',                # Horizontal bars — better for long labels
            color='negative_ratio',
            color_continuous_scale='RdYlGn_r',  # Red (bad) to Green (good)
            title="Top Organizations Mentioned",
            labels={
                'mention_count': 'Mentions',
                'entity': 'Organization',
                'negative_ratio': 'Negative Ratio'
            }
        )
        fig_orgs.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_orgs, use_container_width=True)

    with col_right:
        # ── Scatter: mentions vs negative ratio (risk quadrant chart) ──
        if len(orgs) > 0:
            fig_scatter = px.scatter(
                orgs,
                x='mention_count',
                y='negative_ratio',
                text='entity',
                size='mention_count',
                color='negative_ratio',
                color_continuous_scale='RdYlGn_r',
                title="Prominence vs Negativity (Risk Quadrant)",
                labels={
                    'mention_count': 'Prominence (# Mentions)',
                    'negative_ratio': 'Negativity Ratio',
                }
            )
            fig_scatter.update_traces(textposition='top center', textfont_size=9)
            # Add quadrant lines
            fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ── People in the news ──
    if not people.empty:
        st.subheader("👤 Key People in the News")
        st.dataframe(
            people[['entity', 'mention_count', 'positive_count', 'negative_count', 'negative_ratio']],
            hide_index=True,
            use_container_width=True
        )


# ─────────────────────────────────────────
# SECTION 4: TOPIC MODELING
# ─────────────────────────────────────────

def render_topic_section(df: pd.DataFrame):
    """
    Renders the LDA topic modeling section:
    - Topic distribution across articles
    - Sentiment breakdown per topic
    """
    st.header("🗂️ Topic Analysis (LDA)")

    if 'dominant_topic' not in df.columns:
        st.warning("No topic data found. Run phase5_topic_modeling.py first.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        # ── Topic distribution ──
        topic_counts = df['dominant_topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        topic_counts['Topic'] = 'Topic ' + topic_counts['Topic'].astype(str)

        fig_topics = px.bar(
            topic_counts,
            x='Topic',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            title="Article Distribution by Topic"
        )
        fig_topics.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_topics, use_container_width=True)

    with col_right:
        # ── Sentiment breakdown per topic ──
        topic_sentiment = df.groupby(['dominant_topic', 'finbert_label']).size().reset_index(name='count')
        topic_sentiment['dominant_topic'] = 'Topic ' + topic_sentiment['dominant_topic'].astype(str)

        fig_ts = px.bar(
            topic_sentiment,
            x='dominant_topic',
            y='count',
            color='finbert_label',
            color_discrete_map={
                'positive': '#00C851',
                'negative': '#FF4444',
                'neutral': '#FFBB33'
            },
            title="Sentiment by Topic",
            labels={'dominant_topic': 'Topic', 'count': 'Count', 'finbert_label': 'Sentiment'},
            barmode='stack'
        )
        fig_ts.update_layout(height=350)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Topic confidence ──
    if 'topic_probability' in df.columns:
        avg_prob = df.groupby('dominant_topic')['topic_probability'].mean().reset_index()
        avg_prob.columns = ['Topic', 'Avg Confidence']
        avg_prob['Topic'] = 'Topic ' + avg_prob['Topic'].astype(str)
        avg_prob['Avg Confidence'] = avg_prob['Avg Confidence'].round(3)

        st.subheader("Topic Assignment Confidence")
        st.dataframe(avg_prob, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────
# SECTION 5: ARTICLE BROWSER
# ─────────────────────────────────────────

def render_article_browser(df: pd.DataFrame):
    """
    Renders a searchable, filterable table of all articles with NLP results.
    Great for exploring specific articles in depth.
    """
    st.header("🔍 Article Browser")

    # Sidebar filters
    with st.sidebar:
        st.subheader("🎛️ Filters")

        # Sentiment filter
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )

        # Topic filter
        if 'dominant_topic' in df.columns:
            topic_options = sorted(df['dominant_topic'].dropna().unique().tolist())
            topic_filter = st.multiselect(
                "Filter by Topic",
                options=topic_options,
                default=topic_options,
                format_func=lambda x: f"Topic {int(x)}"
            )
        else:
            topic_filter = None

        # Search box
        search_query = st.text_input("🔎 Search article titles", "")

    # Apply filters to the DataFrame
    filtered_df = df[df['finbert_label'].isin(sentiment_filter)]

    if topic_filter is not None:
        filtered_df = filtered_df[filtered_df['dominant_topic'].isin(topic_filter)]

    if search_query:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False)
        ]

    st.caption(f"Showing {len(filtered_df)} of {len(df)} articles")

    # Display filtered articles
    display_cols = ['title', 'source_name', 'published_at', 'finbert_label',
                    'finbert_confidence', 'vader_label', 'dominant_topic']
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols].head(100),    # Show top 100 filtered results
        hide_index=True,
        use_container_width=True,
        column_config={
            "finbert_label": st.column_config.TextColumn("FinBERT Sentiment"),
            "finbert_confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.2f", min_value=0, max_value=1
            ),
            "published_at": st.column_config.DatetimeColumn("Published", format="DD/MM/YYYY"),
        }
    )


# ─────────────────────────────────────────
# MAIN DASHBOARD APP
# ─────────────────────────────────────────

def main():
    """
    Main function that assembles and renders the entire dashboard.
    Streamlit re-runs this function every time a user interacts with any widget.
    """
    # Load data
    df, entity_df = load_data()

    # Show error if data pipeline hasn't been run yet
    if df is None:
        st.error("⚠️ No data found! Please run the NLP pipeline first:")
        st.code("""
# Run these in order:
python phase1_data_collection.py
python phase2_preprocessing.py
python phase3_sentiment.py
python phase4_ner.py
python phase5_topic_modeling.py

# Then run the dashboard:
streamlit run dashboard.py
        """)
        return

    # Render all sections
    render_header(df)
    render_sentiment_section(df)
    st.divider()
    render_entity_section(entity_df)
    st.divider()
    render_topic_section(df)
    st.divider()
    render_article_browser(df)

    # Footer
    st.divider()
    st.caption("Built with FinBERT (ProsusAI) · spaCy · scikit-learn · Streamlit | NLP Project Portfolio")


# Entry point
if __name__ == "__main__":
    main()
