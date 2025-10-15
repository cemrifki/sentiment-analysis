"""
Morpho-Delta Classifier for Turkish Sentiment Analysis.
Uses Zemberek for morphological analysis and adjusts root word sentiment
based on morpheme multipliers before applying delta-IDF scores.


Author: Cem Rifki Aydin
Date: 15/04/2020

"""

# Standard libraries
import warnings

# Data manipulation
import pandas as pd
import numpy as np

# Java integration
import jpype
import jpype.imports
from jpype.types import *

# Scikit-learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


from src.utils.utils import preprocess_text, compute_delta_idf, LABEL_MAP


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

# Step 3: Parsed texts to sentiment features
def parsed_text_to_three_sent_scores(parsed_text, root_delta):
    polarities = []
    for token, root, morphs in parsed_text:
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

def get_root_only(parsed_texts):
    """
    Extract only root words from parsed texts.
    parsed_texts: list of parsed texts (output of parse_sentence)
    """
    roots = [" ".join([root for token, root, morphs in parsed_tokens]) \
             for parsed_tokens in parsed_texts]
    return roots

def main(args=None):
    """Main execution function"""
    dataset_path=args.dataset
    df = pd.read_csv(dataset_path)  # must have columns: text, sentiment
    
    df[LABEL] = (df[LABEL].str.lower().
                map(LABEL_MAP)
                )

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL], test_size=0.2, stratify=df[LABEL], random_state=42
    )

    # Preprocess  texts
    preprocessed_train = [preprocess_text(text, lang=args.lang) for text in X_train_texts]
    preprocessed_test = [preprocess_text(text, lang=args.lang) for text in X_test_texts]

    # Parse texts with Zemberek
    X_train_parsed = [parse_sentence(text) for text in preprocessed_train]
    X_test_parsed = [parse_sentence(text) for text in preprocessed_test]

    # Delta-IDF scores from training set
    delta_idf_scores = compute_delta_idf(get_root_only(X_train_parsed), y_train, lang="turkish")

    # Extract sentiment features (mean, max, min adjusted scores)
    X_train_features = np.array([parsed_text_to_three_sent_scores(t, delta_idf_scores) for t in X_train_parsed])
    X_test_features = np.array([parsed_text_to_three_sent_scores(t, delta_idf_scores) for t in X_test_parsed])


    # Train SVM
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train_features, y_train)

    # Predictions
    y_pred = clf.predict(X_test_features)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main(args=None)
