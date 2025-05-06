# NeuroEngine: AI Insight Engine

This is a multi-task NLP web application built with Flask and hosted using Render. It supports:
- Amazon Review Sentiment Analysis
- SMS Spam Detection
- COVID-19 Tweet Sentiment
- Fake News Detection
- News Category Classification

## To Run Locally:
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place all model and vectorizer `.pkl` files inside the `/models` folder.
4. Run with: `python app.py`

## Deployment on Render:
- Set build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
