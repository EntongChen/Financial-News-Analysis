import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Bloomberg Financial News Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Models
# ----------------------------
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

# ----------------------------
# Header
# ----------------------------

st.title("📊 Bloomberg Financial News Intelligence Platform")

st.markdown("""
This application supports **investment risk assessment** and **data-driven decision making** by automatically analyzing financial news.

### Analysis Pipeline

**Pipeline 1: Financial Sentiment Analysis**
- Model: Fine-tuned FinBERT
- Output: Bullish / Bearish / Neutral

**Pipeline 2: Financial Topic Classification**
- Model: Fine-tuned DistilBERT
- Output: Earnings, M&A, Macro, Crypto, Markets, etc.
""")

# ----------------------------
# Example News
# ----------------------------

with st.expander("📄 Example Financial News"):

    st.write("""
Apple reported record quarterly earnings, beating analyst expectations on both revenue and profit. The company also raised its guidance for the next quarter, citing strong demand for its new products.
""")

# ----------------------------
# Input
# ----------------------------

news = st.text_area(
    "Paste a financial news article or headline:",
    height=200
)

# ----------------------------
# Analysis
# ----------------------------

if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter financial news text.")
    else:

        sentiment = sentiment_model(news)[0]
        topic = topic_model(news)[0]

        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]

        topic_label = topic["label"]
        topic_score = topic["score"]

        st.divider()

        col1, col2 = st.columns(2)

        # ------------------
        # Sentiment
        # ------------------

        with col1:

            st.subheader("📈 Market Sentiment")

            if sentiment_label.lower() in ["bullish", "positive"]:

                st.success(
                    f"{sentiment_label} ({sentiment_score:.2%})"
                )

                st.progress(float(sentiment_score))

                st.info(
                    "The news reflects positive market sentiment and may indicate favorable expectations for future performance."
                )

            elif sentiment_label.lower() in ["bearish", "negative"]:

                st.error(
                    f"{sentiment_label} ({sentiment_score:.2%})"
                )

                st.progress(float(sentiment_score))

                st.info(
                    "The news reflects negative market sentiment and may signal elevated investment risk."
                )

            else:

                st.warning(
                    f"{sentiment_label} ({sentiment_score:.2%})"
                )

                st.progress(float(sentiment_score))

                st.info(
                    "The news appears neutral and primarily informational."
                )

        # ------------------
        # Topic
        # ------------------

        with col2:

            st.subheader("🏷️ News Topic")

            st.info(
                f"{topic_label} ({topic_score:.2%})"
            )

            st.progress(float(topic_score))
