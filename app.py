import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Financial News Analysis",
    page_icon="📈"
)

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

st.title("📈 Financial News Analysis")

st.write(
    "Analyze financial news using fine-tuned FinBERT models."
)

news = st.text_area(
    "Enter a financial news headline or article:"
)

if st.button("Analyze"):

    if news.strip() == "":
        st.warning("Please enter some text.")
    else:

        sentiment = sentiment_model(news)[0]
        topic = topic_model(news)[0]

        st.subheader("Sentiment")

        st.success(
            f"{sentiment['label']} "
            f"({sentiment['score']:.2%})"
        )

        st.subheader("Topic")

        st.info(
            f"{topic['label']} "
            f"({topic['score']:.2%})"
        )
