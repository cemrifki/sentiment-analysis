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
# Text Preprocessing
# -----------------------------
def preprocess_text(text, lang="english"):
    """Lowercase, remove punctuation, stopwords"""
    text = str(text).lower()
    # text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"([.?!,;:]+)", " \1 ", text)
    text= re.sub(r"[ ]+", " ", text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words(lang)) if lang in stopwords.fileids() else set()
    tokens = [t for t in tokens]  # if t not in stop_words]
    return tokens


# -----------------------------
# Delta-IDF Computation
# -----------------------------
def compute_delta_idf(docs, labels, lang="english"):
    """
    Compute delta-IDF per word based on class distributions.
    labels: "positive"/"negative"
    """
    vectorizer = TfidfVectorizer(tokenizer=lambda x: preprocess_text(x, lang), lowercase=False)
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    # Split by class
    pos_idx = np.where(labels == "positive")[0]
    neg_idx = np.where(labels == "negative")[0]

    # Term frequencies
    pos_tf = np.array(X[pos_idx].sum(axis=0)).flatten() + 1
    neg_tf = np.array(X[neg_idx].sum(axis=0)).flatten() + 1

    # Delta IDF score
    delta_idf = np.log(pos_tf / neg_tf)

    return dict(zip(vocab, delta_idf))


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

    lang=args.lang
    dataset_path=args.dataset

    df = pd.read_csv(dataset_path)  # must have columns: text, sentiment
    results = run_supervised_pipeline(df, lang=lang)

    print(results["report"])


if __name__ == "__main__":
    main()