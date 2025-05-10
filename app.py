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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create an app object using the flask class
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

# Load BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Path to your saved BERT model
bert_model_path = "models"

try:
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        bert_model_path)
    bert_model.to(device)
    bert_model.eval()
    bert_available = True
    print("DistilBERT model loaded successfully!")
except Exception as e:
    print(f"Error loading DistilBERT model: {e}")
    bert_available = False

models = {
    "BERT": None,  # Will be updated if BERT is available
    "XGBoost": xgboost_model,
    "Random Forest": random_forest_model,
    "LightGBM": lightgbm_model,
    "Logistic Regression": logistic_regression_model
}

if bert_available:
    models["BERT"] = bert_model
else:
    models.pop("BERT")


def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def preprocess(text):
    text = clean(text)
    tokens = [portStemmer.stem(word)
              for word in text.split() if word not in stop_words]
    return ' '.join(tokens)


def predict_with_bert(text):
    # Truncate to BERT's max length (typically 512 tokens)
    max_length = 512

    # Tokenize the text
    inputs = bert_tokenizer(text, return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                            padding="max_length")

    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Get the predicted class and confidence
    probs = probabilities.cpu().numpy()[0]
    prediction = np.argmax(probs)
    confidence = float(probs[prediction]) * 100

    # Assuming class 1 is FAKE and class 0 is REAL
    # You may need to adjust this based on your model's training
    label = "FAKE" if prediction == 1 else "REAL"

    print(f"DistilBERT prediction: {label} with confidence {confidence:.2f}%")
    return label, confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    article = request.form['article']

    results, consensus = predict_single_article(article, return_detailed=True)

    # Add description for each model
    descriptions = {
        "BERT": "A state-of-the-art transformer-based language model that understands context and nuance in text.",
        "XGBoost": "An optimized gradient boosting library designed for high performance.",
        "Random Forest": "An ensemble of decision trees trained on random subsets of data.",
        "LightGBM": "A fast, gradient-boosting framework using tree-based learning algorithms.",
        "Logistic Regression": "A simple linear model for binary classification problems."
    }

    # Pass article back to the template so it remains in the textarea
    return render_template('index.html',
                           results=results,
                           consensus=consensus,
                           descriptions=descriptions,
                           article=article)  # Pass the article text back


def predict_single_article(article, return_detailed=False):
    # Preprocess the article before vectorizing
    processed_article = preprocess(article)
    article_vectorized = vectorizer.transform([processed_article])

    model_results = {}

    # First run BERT prediction if available
    if bert_available:
        label, confidence = predict_with_bert(article)
        model_results["BERT"] = {
            "label": label,
            "confidence": confidence
        }

    # Then run other model predictions
    for name, model in models.items():
        if name == "BERT":
            continue  # Skip BERT as it's already processed

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

    # Calculate consensus - BERT counts as 2 votes due to its higher accuracy
    votes = []
    for model_name, info in model_results.items():
        if model_name == "BERT":
            # Count BERT twice
            votes.extend([info['label'], info['label']])
        else:
            votes.append(info['label'])

    consensus_label = "FAKE" if votes.count(
        "FAKE") > len(votes) / 2 else "REAL"

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
