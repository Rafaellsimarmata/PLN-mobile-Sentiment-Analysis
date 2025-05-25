# Pln mobile Sentiment Analysis

This project performs sentiment analysis on user reviews from the **PLN Mobile App** (Indonesian electricity provider). It processes and labels Indonesian text data, then trains **RNN, Random Forest, and SVM** models using **TF-IDF** for feature extraction to classify sentiments into **positive, negative, or neutral**.

## Overview

This project implements a sentiment analysis system that:

- Scrapes reviews from Google Play Store (specifically from the Pln mobile app)
- Processes and labels Indonesian text data
- Trains a RNN model using Embedding feature extraction
- Trains a Random Forest model using TF-Idf feature extraction
- Trains a SVM model using FastText feature extraction
- Implements an easy-to-use inference interface


## Installation

1. Clone the repository:

```bash
git clone https://github.com/Rafaellsimarmata/PLN-mobile-Sentiment-Analysis
cd PLN-mobile-Sentiment-Analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have the model files in the `model/` directory:
   - `sentiment_analysis_model_rnn.h5`
   - `label_encoder.pickle`
   - `tokenizer.pickle`

## Usage

### Quick Start

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Predict sentiment for a text
# Example text to tokenize
text = "Aplikasi sangat bermanfaat dan memudahkan pengguna"

a = sentiment_analyzer(text)
print(f"Text: {text}")
print(f"Prediksi Sentimen: {a}")
```

### Using the Inference Notebook

The project includes a Jupyter notebook for easy inference:

1. Open `notebook_inference.ipynb`
2. Change the `TEXT` variable to your input text
3. Run the notebook cells to get a sentiment prediction

## Dataset

The model was trained on reviews scraped from the PlnMobile app on Google Play Store:

- 20,000 reviews in Indonesian language
- Distribution: ~48% negative, ~42% positive, ~10% neutral
- Initial labeling performed using a pre-trained multilingual sentiment model

## Project Structure

```
├── model/                   # Saved model files
├── notebook_inference.ipynb # Inference notebook
├── notebook_pelatihan_model.ipynb # Training notebook
├── notebook_scrapping.ipynb # Scrapping notebook
├── sentiment_analyzer/      # Python package
│   └── RNN_SentimentAnalyzer.py.py # Sentiment analyzer class
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```