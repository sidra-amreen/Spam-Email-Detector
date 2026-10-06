# Spam Email Detector (Machine Learning)

Classifies messages as **spam** or **ham** using TF-IDF features and a comparison of
three classic models: Multinomial Naive Bayes, Logistic Regression and Linear SVM.
The best model (by spam F1-score) is saved automatically.

## Setup
```bash
pip install -r requirements.txt
python train.py          
python predict.py "Congratulations! You won a free prize, click now"
python predict.py       
```

## How it works
1. `data_loader.py` – downloads the SMS Spam Collection (~5.5k messages); falls back to a tiny built-in sample if offline.
2. `train.py` – 80/20 stratified split, TF-IDF (unigrams + bigrams, stop words removed), trains 3 models, prints precision/recall/F1 and saves a confusion matrix.
3. `predict.py` – loads the saved pipeline and returns SPAM/HAM with a spam score.

## Use your own email data
Put a tab-separated file at `data/spam.tsv` with no header: `label<TAB>text` (label is `ham` or `spam`).

## Ideas to extend
- Add header/URL features, or try a transformer (e.g. DistilBERT)
- Tune with `GridSearchCV`; use cross-validation
- Wrap `predict.py` in a Flask/FastAPI endpoint or Streamlit app
