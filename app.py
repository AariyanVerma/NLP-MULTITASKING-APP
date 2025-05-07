from flask import Flask, request, render_template, jsonify
import joblib
import os

app = Flask(__name__)

# Load models
models = {
    "amazon_sentiment": joblib.load("models/amazon_model.pkl"),
    "sms_spam": joblib.load("models/sms_model.pkl"),
    "corona_sentiment": joblib.load("models/corona_model.pkl"),
    "fake_news": joblib.load("models/fake_model.pkl"),
    "news_category": joblib.load("models/news_model.pkl"),
}

vectorizers = {
    "amazon_sentiment": joblib.load("models/amazon_vectorizer.pkl"),
    "sms_spam": joblib.load("models/sms_vectorizer.pkl"),
    "corona_sentiment": joblib.load("models/corona_vectorizer.pkl"),
    "fake_news": joblib.load("models/fake_vectorizer.pkl"),
    "news_category": joblib.load("models/news_vectorizer.pkl"),
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    task = data["task"]
    text = data["text"]

    model = models.get(task)
    vectorizer = vectorizers.get(task)

    if not model or not vectorizer:
        return jsonify({"result": "Invalid Task Selected"})

    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]

    # 🆕 Post-process label mapping
    if task == "sms_spam":
        label_map_sms = {
            0: "Not Spam", 1: "Spam",     # 🆕 Handles int output
            "0": "Not Spam", "1": "Spam"  # 🆕 Handles str output
        }
        prediction = label_map_sms.get(prediction, prediction)  # 🆕

    elif task == "fake_news":
        label_map_fake = {
            0: "Real", 1: "Fake",         # 🆕
            "0": "Real", "1": "Fake"      # 🆕
        }
        prediction = label_map_fake.get(prediction, prediction)  # 🆕

    elif task == "amazon_sentiment":
        label_map_amazon = {
            0: "Neutral", 1: "Negative", 2: "Positive",     # 🆕
            "0": "Neutral", "1": "Negative", "2": "Positive" # 🆕
        }
        prediction = label_map_amazon.get(prediction, prediction)  # 🆕

    elif task == "news_category":
        label_map_news = {
            1: "Social Issues / Politics",
            2: "Sports",
            3: "Finance / Economy",
            4: "Science & Technology",   # 🆕 All below are mappings
            "1": "Social Issues / Politics",
            "2": "Sports",
            "3": "Finance / Economy",
            "4": "Science & Technology"
        }
        prediction = label_map_news.get(prediction, prediction)  # 🆕

    return jsonify({"result": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
