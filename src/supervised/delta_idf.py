"""
Delta-IDF based Text Classification using SVM.

Author: Cem Rifki Aydin
Date: 16/04/2020

"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Local imports
from src.utils.utils import preprocess_text, compute_delta_idf, LABEL_MAP


# Download stopwords if not already
nltk.download("stopwords")
from nltk.corpus import stopwords


# Sentiment label mapping
LABEL_MAP = {
    "pos": "positive",
    "p": "positive",
    "neg": "negative",
    "n": "negative"
}

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(docs, word_scores, lang="english"):
    """Get min, max, mean polarity scores per document."""
    features = []

    for doc in docs:
        tokens = preprocess_text(doc, lang)
        scores = [word_scores.get(tok, 0.0) for tok in tokens]

        if len(scores) == 0:
            feats = [0.0, 0.0, 0.0]
        else:
            feats = [np.min(scores), np.max(scores), np.mean(scores)]

        features.append(feats)

    return np.array(features)

# -----------------------------
# Training + Evaluation
# -----------------------------
def run_supervised_pipeline(df, lang="english", text_col="text", label_col="sentiment"):
    # Encode labels
    df = df.copy()

    df[label_col] = (df[label_col].str.lower().
                     map(LABEL_MAP)
                     )
    
    # Drop rows with missing labels
    df = df.dropna(subset=[label_col])

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)

    # Delta-IDF scores from training set
    word_scores = compute_delta_idf(train_df[text_col].tolist(), train_df[label_col].values, lang=lang)

    # Extract features
    X_train = extract_features(train_df[text_col].tolist(), word_scores, lang=lang)
    y_train = train_df[label_col].values

    X_test = extract_features(test_df[text_col].tolist(), word_scores, lang=lang)
    y_test = test_df[label_col].values

    # Train SVM
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="positive")
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"])

    return {
        "accuracy": acc,
        "f1": f1,
        "report": report,
        "model": clf,
        "word_scores": word_scores,
    }


def main(args):

    lang = args.lang
    dataset_path = args.dataset

    df = pd.read_csv(dataset_path)  # must have columns: text, sentiment
    results = run_supervised_pipeline(df, lang=lang)

    print(results["report"])


if __name__ == "__main__":
    main()