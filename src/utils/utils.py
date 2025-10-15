# Standard libraries
import re
from collections import Counter

# Data manipulation
import numpy as np

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup / download resources
nltk.download("stopwords", quiet=True)  # download only if needed, suppress output


# Constants
ENGLISH_STOPWORDS = set(stopwords.words("english"))
TURKISH_STOPWORDS = set(stopwords.words("turkish"))

# Sentiment label mapping
LABEL_MAP = {
    "pos": "positive",
    "p": "positive",
    "neg": "negative",
    "n": "negative"
}

def tokenize(text: str, lang='english'):
    """Lowercase + whitespace split + English stopword removal."""
    text = text.lower()
    # Split on whitespace and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    return [t for t in tokens if t not in (ENGLISH_STOPWORDS if lang == 'english' else TURKISH_STOPWORDS)]

def filter_by_frequency(matrix, min_freq=20):
    """
    matrix: list of lists of strings
    min_freq: keep only words with frequency >= min_freq across the whole matrix
    """
    # Flatten and count frequencies
    freq = Counter(word for row in matrix for word in row)

    # Filter each row
    filtered = [[word for word in row if freq[word] >= min_freq] for row in matrix]
    return filtered

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text, lang="english"):
    """Lowercase, remove punctuation, stopwords"""
    text = str(text).lower()
    # text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"([.?!,;:]+)", " \1 ", text)
    text = re.sub(r"[ ]+", " ", text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words(lang)) if lang in stopwords.fileids() else set()
    tokens = [t for t in tokens if t not in stop_words]
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
