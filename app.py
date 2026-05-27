import streamlit as st
from transformers import pipeline

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Financial News Intelligence",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# Load Models
# =====================================================

@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "text-classification",
        model="Nora0211/finbert-sentiment-finetuned"
    )
    topic_model = pipeline(
        "text-classification",
        model="Nora0211/finbert-topic-finetuned"
    )
    return sentiment_model, topic_model


sentiment_model, topic_model = load_models()

# =====================================================
# Session State
# =====================================================

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# =====================================================
# Header
# =====================================================

st.title("📊 Financial News Intelligence")

st.markdown("""
Analyze financial news and generate structured insights for investment decision support.

This application automatically identifies:
- 📈 **Market Sentiment** — Bullish, Bearish, or Neutral
- 🏷️ **Primary Topic** — across 20 financial categories
- 📋 **Interpretation** — a plain-language summary of findings
""")

# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("💡 Quick Examples")

example_news = {
    "Apple Earnings Beat Expectations":
        "Apple reported record quarterly earnings, beating analyst expectations on both revenue and profit. The company also raised its guidance for the next quarter.",
    "Tesla Misses Forecast":
        "Tesla shares fell sharply after the company reported lower-than-expected quarterly earnings and warned of weaker demand in key markets.",
    "Microsoft Acquires AI Startup":
        "Microsoft announced the acquisition of a leading artificial intelligence startup in a deal valued at $12 billion, strengthening its AI portfolio."
}

selected_example = st.sidebar.selectbox(
    "Choose an example:",
    ["None"] + list(example_news.keys())
)

if selected_example != "None":
    st.session_state.news_text = example_news[selected_example]

st.sidebar.divider()
st.sidebar.markdown("**ℹ️ Models Used**")
st.sidebar.markdown("- Sentiment: FinBERT (fine-tuned)")
st.sidebar.markdown("- Topic: FinBERT-tone (fine-tuned)")
st.sidebar.markdown("- Dataset: Twitter Financial News")

# =====================================================
# Input
# =====================================================

news = st.text_area(
    "📰 Paste a financial news article or headline:",
    value=st.session_state.news_text,
    height=250,
    placeholder="Enter financial news here..."
    key="news_input"
)

# =====================================================
# Analyze / Clear Buttons
# =====================================================

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    analyze = st.button("🔍 Analyze News", use_container_width=True)

with col_btn2:
    if st.button("🗑️ Clear", use_container_width=True):
    st.session_state.news_text = ""
    st.session_state.news_input = ""    # 加这行
    st.rerun()

# =====================================================
# Analysis
# =====================================================

if analyze:

    if news.strip() == "":
        st.warning("Please enter a financial news article.")
        st.stop()

    if len(news.split()) < 5:
        st.warning("Please enter a longer text (at least 5 words) for better accuracy.")
        st.stop()

    with st.spinner("Analyzing news..."):
        sentiment    = sentiment_model(news)[0]
        topic_result = topic_model(news, top_k=1)[0]

    sentiment_label = sentiment["label"]
    sentiment_score = sentiment["score"]
    primary_topic   = topic_result["label"]
    primary_score   = topic_result["score"]

    st.divider()

    # =================================================
    # Main Results
    # =================================================

    col1, col2 = st.columns(2)

    # -----------------------------
    # Sentiment
    # -----------------------------

    with col1:
        st.subheader("📈 Market Sentiment")

        sentiment_icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
        icon = sentiment_icons.get(sentiment_label, "⚪")

        st.metric(
            label="Sentiment",
            value=f"{icon} {sentiment_label}"
        )
        st.write(f"Confidence: {sentiment_score:.1%}")
        st.progress(float(sentiment_score))

    # -----------------------------
    # Primary Topic
    # -----------------------------

    with col2:
        st.subheader("🏷️ Primary Topic")

        st.metric(
            label="Topic",
            value=primary_topic
        )
        st.write(f"Confidence: {primary_score:.1%}")
        st.progress(float(primary_score))

    # =================================================
    # Interpretation
    # =================================================

    st.divider()
    st.subheader("📋 Interpretation")

    st.write(f"""
The article is classified as **{sentiment_label}** in market sentiment.

The dominant topic is **{primary_topic}**, with a confidence score of **{primary_score:.1%}**.

Overall, the news reflects a **{sentiment_label.lower()}** market tone and is primarily associated with **{primary_topic}**.
    """)

    # =================================================
    # Article Statistics
    # =================================================

    st.divider()

    word_count = len(news.split())
    char_count = len(news)

    col3, col4 = st.columns(2)
    with col3:
        st.metric("📝 Words", word_count)
    with col4:
        st.metric("🔤 Characters", char_count)

# =====================================================
# Footer
# =====================================================

st.divider()
st.caption("Financial News Intelligence Platform | Sentiment Analysis and Topic Classification")
