# Standard libraries
import re
from collections import Counter
import string

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

def filter_by_frequency(matrix, min_freq=3):
    """
    matrix: list of lists of strings
    min_freq: keep only words with frequency >= min_freq across the whole matrix
    """
    # Flatten and count frequencies
    freq = Counter(word for row in matrix for word in row)

    # Filter each row
    filtered = [[word for word in row if freq[word] >= min_freq] for row in matrix]
    return filtered


# For supervised approaches, leveraging the below one led to better success rates
def preprocess_text(text, lang='english'):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text, language=lang)
    return " ".join(tokens)