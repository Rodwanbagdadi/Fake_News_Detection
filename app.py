from flask import Flask, render_template, request, redirect
import pickle
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import joblib

# Creat an app object using the flask class
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
portStemmer = PorterStemmer()

# Load models and vectorizer once
xgboost_model = joblib.load("models/xgboost_model.pkl")
random_forest_model = joblib.load("models/random_forest_model.pkl")
lightgbm_model = joblib.load("models/light_gbm_model.pkl")
logistic_regression_model = joblib.load("models/logistic_regression_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

models = {
    "XGBoost": xgboost_model,
    "Random Forest": random_forest_model,
    "LightGBM": lightgbm_model,
    "Logistic Regression": logistic_regression_model
}


def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def preprocess(text):
    text = clean(text)
    tokens = [portStemmer.stem(word)
              for word in text.split() if word not in stop_words]
    return ' '.join(tokens)


@app.route('/')
def home():
    return render_template('index.html')

# In your predict() function inside app.py


@app.route('/predict', methods=['POST'])
def predict():
    article = request.form['article']

    results, consensus = predict_single_article(article, return_detailed=True)

    # Add description for each model
    descriptions = {
        "XGBoost": "An optimized gradient boosting library designed for high performance.",
        "Random Forest": "An ensemble of decision trees trained on random subsets of data.",
        "LightGBM": "A fast, gradient-boosting framework using tree-based learning algorithms.",
        "Logistic Regression": "A simple linear model for binary classification problems."
    }

    return render_template('index.html', results=results, consensus=consensus, descriptions=descriptions)


def predict_single_article(article, return_detailed=False):
    article_vectorized = vectorizer.transform([article])

    model_results = {}

    for name, model in models.items():
        prob = model.predict_proba(article_vectorized)[0]
        prediction = model.predict(article_vectorized)[0]
        confidence = max(prob) * 100
        label = "FAKE" if prediction == 1 else "REAL"
        model_results[name] = {
            "label": label,
            "confidence": confidence
        }

    # Sort by confidence descending
    model_results = dict(sorted(model_results.items(),
                         key=lambda item: item[1]['confidence'], reverse=True))

    # Calculate consensus
    votes = [info['label'] for info in model_results.values()]
    consensus_label = "FAKE" if votes.count("FAKE") >= 3 else "REAL"

    if return_detailed:
        return model_results, consensus_label
    else:
        return {
            "article": article,
            "results": model_results,
            "consensus": consensus_label
        }


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False)
