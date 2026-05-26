import streamlit as st
from transformers import pipeline

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Financial News Intelligence",
    page_icon="📊",
    layout="centered"
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
# Header
# =====================================================

st.title("📊 Financial News Intelligence")

st.markdown("""
Analyze financial news and generate structured interpretation for investment decision support.

This application automatically identifies:

- 📈 Market Sentiment
- 🏷️ News Topic
- 📋 Interpretation
""")

# =====================================================
# Example News
# =====================================================

with st.expander("📄 Example Financial News"):

    st.write("""
Apple reported record quarterly earnings, beating analyst expectations on both revenue and profit. The company also raised its guidance for the next quarter, citing strong demand for its new products.
""")

# =====================================================
# User Input
# =====================================================

news = st.text_area(
    "Paste a financial news article or headline:",
    height=220,
    placeholder="Enter financial news here..."
)

# =====================================================
# Analysis
# =====================================================

if st.button("Analyze News", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter a financial news article.")
        st.stop()

    with st.spinner("Analyzing news..."):

        sentiment = sentiment_model(news)[0]
        topic_results = topic_model(
    news,
    top_k=3
)
        for i, result in enumerate(topic_results, start=1):
    st.write(
        f"{i}. {result['label']} ({result['score']:.1%})"
    )

    sentiment_label = sentiment["label"]
    sentiment_score = sentiment["score"]

    topic_label = topic["label"]
    topic_score = topic["score"]

    st.divider()

    # =================================================
    # Results Section
    # =================================================

    col1, col2 = st.columns(2)

    # -------------------------------
    # Sentiment
    # -------------------------------

    with col1:

        st.subheader("📈 Market Sentiment")

        if sentiment_label.lower() in ["bullish", "positive"]:

            st.success(sentiment_label)

        elif sentiment_label.lower() in ["bearish", "negative"]:

            st.error(sentiment_label)

        else:

            st.warning(sentiment_label)

        st.write(f"**Confidence:** {sentiment_score:.1%}")
        st.progress(float(sentiment_score))

    # -------------------------------
    # Topic
    # -------------------------------

    with col2:

        st.subheader("🏷️ News Topic")

        st.info(topic_label)

        st.write(f"**Confidence:** {topic_score:.1%}")
        st.progress(float(topic_score))

    # =================================================
    # Interpretation
    # =================================================

    st.divider()

    st.subheader("📋 Interpretation")

    st.write(
        f"""
The submitted article is classified as **{sentiment_label}** in market sentiment and belongs to the **{topic_label}** topic category.

The sentiment classification was generated with a confidence score of **{sentiment_score:.1%}**, while the topic classification achieved a confidence score of **{topic_score:.1%}**.

This result suggests that the news primarily relates to **{topic_label}** and reflects a **{sentiment_label.lower()}** market tone.
"""
    )

# =====================================================
# Footer
# =====================================================

st.divider()

st.caption(
    "Financial News Intelligence Platform | Financial News Sentiment Analysis and Topic Classification"
)
