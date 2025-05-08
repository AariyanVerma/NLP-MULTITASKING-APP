import streamlit as st
import joblib

# Load models
models = {
    "Amazon Sentiment": joblib.load("models/amazon_model.pkl"),
    "SMS Spam": joblib.load("models/sms_model.pkl"),
    "Corona Sentiment": joblib.load("models/corona_model.pkl"),
    "Fake News": joblib.load("models/fake_model.pkl"),
    "News Category": joblib.load("models/news_model.pkl")
}

# Load vectorizers
vectorizers = {
    "Amazon Sentiment": joblib.load("models/amazon_vectorizer.pkl"),
    "SMS Spam": joblib.load("models/sms_vectorizer.pkl"),
    "Corona Sentiment": joblib.load("models/corona_vectorizer.pkl"),
    "Fake News": joblib.load("models/fake_vectorizer.pkl"),
    "News Category": joblib.load("models/news_vectorizer.pkl")
}

# Label mappings (int and str handled)
label_maps = {
    "SMS Spam": {
        0: "Not Spam", 1: "Spam",
        "0": "Not Spam", "1": "Spam"
    },
    "Fake News": {
        0: "Real", 1: "Fake",
        "0": "Real", "1": "Fake"
    },
    "Amazon Sentiment": {
        0: "Neutral", 1: "Negative", 2: "Positive",
        "0": "Neutral", "1": "Negative", "2": "Positive"
    },
    "News Category": {
        1: "Social Issues / Politics",
        2: "Sports",
        3: "Finance / Economy",
        4: "Science & Technology",
        "1": "Social Issues / Politics",
        "2": "Sports",
        "3": "Finance / Economy",
        "4": "Science & Technology"
    }
}

# Streamlit App UI
st.set_page_config(page_title="NeuroEngine", layout="centered")
st.title("🧠 NeuroEngine: Multi-Task NLP Analyzer")

task = st.selectbox("Choose a Task", list(models.keys()))
text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        model = models[task]
        vectorizer = vectorizers[task]

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        label_map = label_maps.get(task)
        if label_map:
            pred = label_map.get(pred, pred)

        st.success(f"**Prediction:** {pred}")
