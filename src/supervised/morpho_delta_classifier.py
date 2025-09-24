"""
Morpho-Delta Classifier for Turkish Sentiment Analysis.
Uses Zemberek for morphological analysis and adjusts root word sentiment
based on morpheme multipliers before applying delta-IDF scores.


Author: Cem Rifki Aydin
Date: 15/04/2020

"""

# Standard libraries
import math
import re
import warnings
from collections import defaultdict

# Data manipulation
import pandas as pd
import numpy as np

# Java integration
import jpype
import jpype.imports
from jpype.types import *

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Setup / configuration
warnings.filterwarnings("ignore")

# Column names in dataset
TEXT_COL, LABEL = "text", "sentiment"


# Step 1: Define morpheme multipliers based on linguistic intuition
# These can be tuned based on validation data
# Values <1.0 reduce intensity, >1.0 increase intensity, negative flips polarity
MORPH_MULTIPLIERS = {
    "neg": -1.0,         # flip polarity
    "imp": 1.35,         # mild intensifier
    "opt": 0.85,         # optative -> soften
    "cond": 1.25,        # conditional -> more definitive
    "narr": 0.75,        # narrative/evidential -> soften
    "narrpart": 0.75,    # narrative participle -> soften
    "past": 0.95,        # slight downweight
    "pres": 1.0,
    "prog1": 1.0,
    "prespart": 1.0,
    "aor": 1.0,
    "inf1": 0.80,        # nominalization -> reduces immediacy
    "pass": 0.95,
    "caus": 1.0,
    "ness": 0.70,        # -ness / nominalization -> reduces direct sentiment
    "adj": 1.3,
    "noun": 1.0,
    "verb": 1.3,
    "adv": 1.3,
    "acc": 0.95,
    "dat": 0.95,
    "abl": 0.95,
    "loc": 0.95,
    "gen": 0.95,
    "a1sg": 1.0,
    "p1sg": 1.0,
    "p3sg": 1.0,
    "a3sg": 1.0,
    "a3pl": 1.0,
    "a2pl": 1.0,
    "zero": 1.0,
    "cop": 1.0,
    "agt": 1.0,
    "pres": 1.0,
    "prog1": 1.0,
    # keep any others neutral by default
}


# --- Step 2: Connect to Zemberek ---

# Path to Zemberek jar
ZEMBEREK_PATH = "resources/zemberek-full.jar"

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[ZEMBEREK_PATH], convertStrings=True)

from zemberek.morphology import TurkishMorphology

morphology = TurkishMorphology.createWithDefaults()

# --- Parse sentence into root + morphemes ---
def parse_sentence(sentence):
    tokens = sentence.split()
    results = []
    for token in tokens:
        analysis = morphology.analyzeSentence(token)
        disamb = morphology.disambiguate(token, analysis).bestAnalysis()
        if disamb.size() > 0:
            root = disamb[0].getLemmas()[0]
            morphs = [str(m).split(":")[1].lower() for m in disamb[0].getMorphemes()][1:]

            results.append([token, root, morphs]) 
        else:
            results.append([token, token, []])
    return results

# parse_sentence("Onu pek sevmedim. O güzeldir bayağı. Ayşe oraya mı gitmiş peki?")

def adjust_polarity(root_score, morphemes):
    polarity = root_score
    for m in morphemes:
        polarity *= MORPH_MULTIPLIERS.get(m, 1)        
    return polarity


# Step 3: Texts to features
def text_to_features(text, root_delta):
    parsed = parse_sentence(text)
    polarities = []
    for token, root, morphs in parsed:
        root_score = root_delta.get(root, 0)
        adjusted = adjust_polarity(root_score, morphs)
        if (root in ("yok", "değil")) and polarities:
            polarities[-1] *= -1
            continue
        polarities.append(adjusted)
    if polarities:
        return [np.mean(polarities), np.max(polarities), np.min(polarities)]
    else:
        return [0.0, 0.0, 0.0]


def main(args=None):

    dataset_path=args.dataset
    df = pd.read_csv(dataset_path).iloc[:1000] # must have columns: text, sentiment
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL], test_size=0.2, stratify=df[LABEL], random_state=42
    )

    # Vectorizer + delta-IDF computation
    vectorizer = TfidfVectorizer(lowercase=False)
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)
    terms = vectorizer.get_feature_names_out()

    # Root delta-IDF
    pos_idx = [i for i, l in enumerate(y_train) if l == "P"]
    neg_idx = [i for i, l in enumerate(y_train) if l == "N"]
    root_delta = defaultdict(float)
    for term in terms:
        tidx = vectorizer.vocabulary_[term]
        pos_val = np.sum(X_train_vec[pos_idx, tidx].toarray()) if pos_idx else 0
        neg_val = np.sum(X_train_vec[neg_idx, tidx].toarray()) if neg_idx else 0
        root_delta[term] = math.log((pos_val + 0.001) / (neg_val + 0.001))

    X_train_features = np.array([text_to_features(t, root_delta) for t in X_train_texts])
    X_test_features = np.array([text_to_features(t, root_delta) for t in X_test_texts])

    clf = LogisticRegression()
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main(args=None)
