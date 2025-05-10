# AI-Powered Fake News Detection System

A robust fake news detection system that combines multiple machine learning models, including DistilBERT, to analyze and classify news articles as real or fake.

![Fake News Detection](https://img.shields.io/badge/Fake%20News%20Detection-AI%20Powered-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)

## 🌟 Features

- **Multiple Model Analysis**: Combines predictions from:
  - DistilBERT (Transformer-based)
  - XGBoost
  - Random Forest
  - LightGBM
  - Logistic Regression
- **Interactive Web Interface**: Modern, responsive UI with dark/light mode
- **Detailed Analysis**: Shows confidence scores and reasoning for each model
- **Consensus Voting**: Weighted voting system for final prediction
- **Visual Analytics**: Interactive charts showing model confidence levels

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for BERT model)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using conda
   conda create -n BERT python=3.8
   conda activate BERT

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - Enter a news article in the text area
   - Click "Analyze Article" to get predictions

## 📁 Project Structure
fake-news-detection/
├── app.py                 # Main Flask application
├── models/                # Trained model files
│   ├── model.safetensors  # DistilBERT model
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── ...
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── static/                # Static files (CSS, JS)
└── requirements.txt       # Project dependencies

## 🛠️ Technical Details

### Models Used

1. **DistilBERT**
   - Transformer-based model for deep semantic understanding
   - Handles contextual analysis and nuanced language patterns

2. **Traditional ML Models**
   - XGBoost: Gradient boosting for structured data
   - Random Forest: Ensemble of decision trees
   - LightGBM: Light gradient boosting machine
   - Logistic Regression: Linear classification

### How It Works

1. **Text Preprocessing**
   - Lowercase conversion
   - Special character removal
   - Stopword removal
   - Stemming

2. **Model Prediction**
   - Each model analyzes the preprocessed text
   - BERT model uses transformer architecture
   - Other models use TF-IDF features

3. **Consensus Voting**
   - BERT predictions count double (weighted voting)
   - Final decision based on majority vote

## 🎯 Usage Example

```python
# Example article
article = """
[Your news article text here]
"""

# Get predictions
results, consensus = predict_single_article(article)
print(f"Consensus: {consensus}")
for model, prediction in results.items():
    print(f"{model}: {prediction['label']} ({prediction['confidence']}%)")
```

## 📊 Performance

- **Accuracy**: Varies by model (BERT typically highest)
- **Speed**: 
  - BERT: ~1-2 seconds per article
  - Other models: < 1 second per article
- **Memory Usage**: ~500MB-1GB (mainly due to BERT model)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the DistilBERT model
- Flask for the web framework
- All other open-source libraries used in this project

## 📧 Contact

For questions or suggestions, please open an issue or contact [rodwanbagdadi@gmail.com]

---
Made with ❤️ by Rodwan
