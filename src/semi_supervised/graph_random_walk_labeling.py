"""
Graph-based Semi-Supervised Learning via Random Walk Labeling (SentProp).

Author: Cem Rifki Aydin
Date: 10/01/2020

"""

# ------------------------------
# Standard library imports
# ------------------------------
import json
import random
import math
from collections import Counter, defaultdict
import warnings

# ------------------------------
# Third-party imports
# ------------------------------
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ------------------------------
# Local imports
# ------------------------------
from src.utils.utils import tokenize, filter_by_frequency, LABEL_MAP
from src.utils.seeds import POSITIVE_SEEDS, NEGATIVE_SEEDS

# ------------------------------
# Setup / configuration
# ------------------------------
# Ignore warnings globally
warnings.filterwarnings("ignore")

LANG = "english"


def classify_whole_texts(df, text_col, label_col, word_scores, threshold=0.0):
    """
    Classify texts based on SENTPROP word sentiment scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with text and labels
    text_col : str
        Column containing the text
    label_col : str
        Column containing the gold sentiment labels ("positive"/"negative")
    word_scores : dict
        Dictionary {word: sentiment_score}
    threshold : float
        Decision boundary (default: 0 → positive if score > 0)

    Returns
    -------
    df : pandas.DataFrame
        Original dataframe with new columns: "pred_score", "pred_label"
    metrics : dict
        Accuracy and F1 scores
    """

    pred_scores = []
    pred_labels = []

    for text in df[text_col]:
        tokens = tokenize(text, LANG)  # str(text).lower().split()

        # Sum up sentiment scores for tokens that are in lexicon
        score = np.mean([word_scores.get(tok, 0.0) for tok in tokens])

        pred_scores.append(score)

        pred_labels.append("positive" if score > threshold else "negative")

    # Add predictions to dataframe
    df = df.copy()
    df["pred_score"] = pred_scores
    df["pred_label"] = pred_labels
    gold = df[label_col].str.lower() 

    # Compute metrics
    acc = accuracy_score(gold, df["pred_label"])
    f1 = f1_score(gold, df["pred_label"], pos_label="positive", average="macro")

    report = classification_report(gold, df["pred_label"], target_names=["negative", "positive"])

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "classification_report": report
    }

    return df, metrics


class CorpusEmbeddings:
    """
    Build word embeddings from corpus using PPMI + SVD.
    """
    def __init__(self, window_size: int = 4, smoothing: float = 0.75, dim: int = 300):
        self.window_size = window_size
        self.smoothing = smoothing
        self.dim = dim

    def build_vocab(self, corpus: list):
        """Build vocabulary from corpus."""
        tokens = [word for doc in corpus for word in tokenize(doc, LANG)]
        vocab = list(set(tokens))
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        return vocab

    def build_cooccurrence(self, corpus: list):
        """Build co-occurrence matrix with sliding window."""
        V = len(self.word_to_idx)
        counts = defaultdict(float)
        total_count = 0

        for doc in corpus:
            words = tokenize(doc, LANG)  # doc.split()
            for i, w in enumerate(words):
                if w not in self.word_to_idx:
                    continue
                w_idx = self.word_to_idx[w]
                left = max(0, i - self.window_size)
                right = min(len(words), i + self.window_size + 1)
                for j in range(left, right):
                    if i == j or words[j] not in self.word_to_idx:
                        continue
                    c_idx = self.word_to_idx[words[j]]
                    counts[(w_idx, c_idx)] += 1.0
                    total_count += 1

        return counts, total_count

    def build_ppmi_matrix(self, corpus: list):
        """Construct smoothed PPMI matrix."""
        counts, total_count = self.build_cooccurrence(corpus)
        V = len(self.word_to_idx)
        word_freq = Counter()
        for (w, c), v in counts.items():
            word_freq[w] += v
            word_freq[c] += v

        # smoothed probabilities
        word_prob = {w: (f ** self.smoothing) for w, f in word_freq.items()}
        norm = sum(word_prob.values())
        for w in word_prob:
            word_prob[w] /= norm

        M = np.zeros((V, V))
        for (w, c), v in counts.items():
            p_wc = v / total_count
            p_w = word_prob[w]
            p_c = word_prob[c]
            val = max(math.log((p_wc / (p_w * p_c)) + 1e-10), 0)
            M[w, c] = val

        return M

    def compute_embeddings(self, corpus: list):
        """Get embeddings from truncated SVD of PPMI matrix."""
        M = self.build_ppmi_matrix(corpus)
        U, _, _ = svds(M, k=min(self.dim, M.shape[0] - 1))
        return U


class SentProp:
    """
    SentProp: Graph-based Random Walk Labeling for Sentiment Induction.
    Reference: Hamilton et al. (2016). Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora.
    """
    def __init__(self, vocab: list, embeddings: np.ndarray, k_neighbors: int = 10, beta: float = 0.85):
        self.vocab = vocab
        self.embeddings = embeddings
        self.k_neighbors = k_neighbors
        self.beta = beta
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        self.graph = None

    def build_graph(self):
        """Lexical graph with cosine similarity."""
        V = len(self.vocab)
        E = np.zeros((V, V))
        for i in range(V):
            sims = []
            for j in range(V):
                if i == j:
                    continue
                sim = 1 - np.dot(self.embeddings[i], self.embeddings[j]) / (
                    np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j]) + 1e-10
                )
                sims.append((j, 1 - sim))  # use similarity
            sims.sort(key=lambda x: x[1], reverse=True)
            for j, sim in sims[:self.k_neighbors]:
                E[i, j] = np.arccos(np.clip(sim, -1, 1))
        self.graph = (E + E.T) / 2

    def propagate(self, seed_words: list):
        """Random walk label propagation."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph().")
        V = len(self.vocab)
        p = np.ones(V) / V
        s = np.zeros(V)
        for word in seed_words:
            if word in self.word_to_idx:
                s[self.word_to_idx[word]] = 1 / len(seed_words)

        D = np.diag(np.sum(self.graph, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D + np.eye(D.shape[0]) * 1e-10))
        T = D_inv_sqrt @ self.graph @ D_inv_sqrt

        tol = 1e-6
        p_prev = np.zeros(V)
        max_iter = 0
        while np.linalg.norm(p - p_prev, 1) > tol:
            if max_iter >= 120:
                break
            max_iter += 1
            p_prev = p.copy()
            p = self.beta * T @ p + (1 - self.beta) * s
        return p

    def score_words(self, positive_seeds: list, negative_seeds: list):
        p_pos = self.propagate(positive_seeds)
        p_neg = self.propagate(negative_seeds)
        raw_scores = p_pos / (p_pos + p_neg + 1e-10)
        standardized_scores = (raw_scores - np.mean(raw_scores)) / np.std(raw_scores)
        return {self.idx_to_word[i]: standardized_scores[i] for i in range(len(self.vocab))}

    def bootstrap_scores(self, positive_seeds: list, negative_seeds: list, B: int = 50, subset_size: int = 7):
        all_scores = []
        for _ in range(B):
            pos_subset = random.sample(positive_seeds, min(subset_size, len(positive_seeds)))
            neg_subset = random.sample(negative_seeds, min(subset_size, len(negative_seeds)))
            scores = self.score_words(pos_subset, neg_subset)
            all_scores.append(scores)

        mean_scores = {}
        std_scores = {}
        for w in self.vocab:
            vals = [b[w] for b in all_scores]
            mean_scores[w] = np.mean(vals)
            std_scores[w] = np.std(vals)
        return mean_scores, std_scores

def make_balanced(df, label_col="sentiment", random_state=42):
    # Normalize labels
    labels = df[label_col].str.lower()
    
    # Find smallest class size
    min_size = labels.value_counts().min()
    
    # Sample equally from each class
    balanced_parts = [
        df[labels == cls].sample(min_size, random_state=random_state)
        for cls in labels.unique()
    ]
    
    return pd.concat(balanced_parts).sample(frac=1, random_state=random_state)


def main(args=None):
    """
    Main function to run the semi-supervised graph-based sentiment labeling.
    """
    
    dataset_path = args.dataset
    LANG = args.lang

    df = pd.read_csv(dataset_path) # must have columns: text, sentiment

    df["sentiment"] = (df["sentiment"].str.lower().
                     map(LABEL_MAP)
                     )
    
    balanced_df = make_balanced(df, label_col="sentiment", random_state=42)

    label_col = "sentiment"
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df[label_col], random_state=42)

    # If you just want the text column as a list
    corpus = train_df["text"].tolist()
    labels = train_df["sentiment"].tolist()

    corpus = filter_by_frequency([tokenize(text, LANG) for text in corpus])
    corpus = [" ".join(text) for text in corpus]

    # Step 1: Build embeddings from corpus
    emb_builder = CorpusEmbeddings(window_size=5, dim=80)
    vocab = emb_builder.build_vocab(corpus)
    embeddings = emb_builder.compute_embeddings(corpus)

    # Step 2: Initialize SentProp
    sentprop = SentProp(vocab, embeddings, k_neighbors=5, beta=0.9)
    sentprop.build_graph()

    # Step 3: Compute scores
    word_scores = sentprop.score_words(POSITIVE_SEEDS, NEGATIVE_SEEDS)
    print("Polarity Scores:")
    print(json.dumps(word_scores, ensure_ascii=False, indent=2))

    # Step 4: Bootstrap for confidence
    mean_scores, std_scores = sentprop.bootstrap_scores(POSITIVE_SEEDS, NEGATIVE_SEEDS, B=20, subset_size=3)
    print("\nBootstrap Mean Scores:")
    print(json.dumps(mean_scores, ensure_ascii=False, indent=2))
    print("\nBootstrap Std Scores:")
    print(json.dumps(std_scores, ensure_ascii=False, indent=2))

    # Classify
    classified_df, metrics = classify_whole_texts(test_df, text_col="text", label_col="sentiment", word_scores=word_scores)
    
    print("\nTraining Set Classification Metrics:")
    print(metrics["classification_report"])    
    print("=========" * 10)


if __name__ == "__main__":
    main(args=None)