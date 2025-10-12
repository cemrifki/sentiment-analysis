"""
Search-Based Unsupervised Label Extraction using Yandex Search Engine.

Author: Cem Rifki Aydin
Date: 12/01/2020

"""

# ------------------------------
# Standard library imports
# ------------------------------
import os
import json
import math
import re
import threading
import time
from typing import Dict, List

# ------------------------------
# Third-party imports
# ------------------------------
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ------------------------------
# Local imports
# ------------------------------
from src.utils.utils import tokenize, filter_by_frequency, LABEL_MAP

# ----------------------
# Helper classes
# ----------------------
class BasicFunctions:
    @staticmethod
    def elapsed_time(start: float, end: float) -> str:
        return f"{(end - start) / 1000:.2f} seconds"

class Constants:
    """Holds constants and configurations.
    WORD_PAIRS: List[List[str]]
        List of positive-negative word pairs for sentiment scoring.
    PRINT_RESULTS: bool
        Whether to print intermediate results.
    """
    WORD_PAIRS_MAP = {
        "turkish": [
            ["sevgi", "nefret"], ["güzel", "çirkin"], ["doğru", "yanlış"],
            ["harika", "berbat"], ["tatlı", "acı"], ["olumlu", "olumsuz"],
            ["zevkli", "sıkıcı"], ["iyi", "kötü"], ["başyapıt", "vasat"]
        ],
        "english": [
            ["love", "hate"], ["beautiful", "ugly"], ["right", "wrong"],
            ["awesome", "terrible"], ["sweet", "bitter"], ["positive", "negative"],
            ["fun", "boring"], ["good", "bad"], ["masterpiece", "mediocre"]
        ]
    }

    def __init__(self, lang="english", print_results=True):
        if lang not in self.WORD_PAIRS_MAP:
            raise ValueError(f"Unsupported language: {lang}. Choose from {list(self.WORD_PAIRS_MAP.keys())}")
        
        self.WORD_PAIRS = self.WORD_PAIRS_MAP[lang]
        self.PRINT_RESULTS = print_results

# ----------------------
# Search Engine Class
# ----------------------
class SearchEngine:
    """
    Search Engine for Unsupervised Label Extraction using Yandex.
    Uses predefined positive-negative word pairs to compute sentiment scores.
    """

    def __init__(self, lang: str, near: int = 4, time_conn: int = 5, cache_path="cached_sentiments.json"):
        self.constants = Constants(lang=lang)
        self.lang = lang
        self.near = near  # <-- configurable NEAR window size
        self.time_conn = time_conn
        self.cache_path = cache_path
        self.cache = self._load_cache()

    @staticmethod
    def str_to_db_yan(s: str) -> float:
        """Convert Yandex count string to float."""
        parts = s.split()
        if not parts:
            return 1.0
        try:
            val = float(parts[0].replace('.', ''))
        except ValueError:
            val = 1.0
        multiplier = 1
        for num_str, mult in zip(["bin", "milyon", "milyar"], [1000., 1000000., 1000000000.]):
            if num_str in s:
                multiplier = mult
                break
        return val * multiplier

    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached sentiments if available."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Persist current cache to disk."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def query_result(self, query_word: str, related_word: str) -> str:
        """Query Yandex and return the number of results as string."""
        near = f"%2F(-{self.near}+%2B{self.near})"
        space = "+"
        all_query = f'{query_word}{space}{near}{space}{related_word}'
        all_query = "\"" + all_query + "\""
        url = f"https://www.yandex.com.tr/search/?text={all_query}&lr=11508" if related_word \
            else f"https://www.yandex.com.tr/search/?text={query_word}&lr=11508"
        try:
            response = requests.get(url, timeout=self.time_conn)
            text = response.text
            if "tam karşılı" in text:
                return "0.22"
            match = re.search(r'Yandex: (.*?) sonuç', text)
            return match.group(1).replace("\u00A0", " ") if match else "1"
        except Exception:
            return "1"

    def calc_pair_score(self, target_word: str, pair: List[str], results_list: List[Dict]):
        """Calculate sentiment score for target word with respect to a word pair."""
        pos_word, neg_word = pair

        # Co-occurrence with positive and negative words
        pos_co = self.query_result(target_word, pos_word)
        neg_co = self.query_result(target_word, neg_word)

        pos_co_parsed = float(self.str_to_db_yan(pos_co)) 
        neg_co_parsed = float(self.str_to_db_yan(neg_co))

        # Prior frequencies (unigram counts)
        pos_prior = self.query_result(pos_word, "")
        neg_prior = self.query_result(neg_word, "")

        pos_prior_parsed = float(self.str_to_db_yan(pos_prior))
        neg_prior_parsed = float(self.str_to_db_yan(neg_prior))

        # Apply Laplace smoothing and prior normalization
        pos_norm = (pos_co_parsed + 1) / (pos_prior_parsed + 1)
        neg_norm = (neg_co_parsed + 1) / (neg_prior_parsed + 1)

        score = math.log10(pos_norm / neg_norm)

        result_obj = {
            "target_word": target_word,
            "positive_word": pos_word,
            "negative_word": neg_word,
            "positive_co": pos_co,
            "negative_co": neg_co,
            "positive_prior": pos_prior,
            "negative_prior": neg_prior,
            "score": score
        }

        results_list.append(result_obj)

    def compute_sentiment(self, target_word: str) -> Dict:
        """Compute or fetch sentiment of the target word."""
        # Return cached result if available
        if target_word in self.cache:
            return self.cache[target_word]

        threads = []
        results_list: List[Dict] = []

        for pair in self.constants.WORD_PAIRS:
            t = threading.Thread(target=self.calc_pair_score, args=(target_word, pair, results_list))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if not results_list:
            result = {"target_word": target_word, "pair_results": [], "average_score": 0.0}
        else:
            average_score = sum(r["score"] for r in results_list) / len(results_list)

            result = {
                "target_word": target_word,
                "pair_results": results_list,
                "average_score": average_score
            }

        # Cache and save incrementally
        self.cache[target_word] = result
        self._save_cache()

        return result

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

def main(args):
    """Main function to run the search-based unsupervised label extraction with global normalization 
    based on average scores."""

    dataset_path = args.dataset
    df = pd.read_csv(dataset_path)
    df["sentiment"] = df["sentiment"].str.lower().map(LABEL_MAP)
    
    # Balanced subset (optional, for fair testing)
    balanced_df = make_balanced(df, label_col="sentiment", random_state=42)

    corpus = balanced_df["text"].tolist()
    labels = balanced_df["sentiment"].tolist()

    # Tokenize and filter
    corpus = filter_by_frequency([tokenize(text, args.lang) for text in corpus])
    corpus = [" ".join(text) for text in corpus]

    # Initialize SearchEngine with caching
    search_engine = SearchEngine(args.lang, args.near if args.near else 4)

    # -------------------------------------------------------
    # Pass 1: Compute sentiment scores for all unique tokens
    # -------------------------------------------------------
    all_tokens = set(" ".join(corpus).split())
    print(f"\n[INFO] Unique tokens to process: {len(all_tokens)}\n")

    token_avg_scores = []

    for tok in tqdm(all_tokens, desc="Caching sentiment scores"):
        result = search_engine.compute_sentiment(tok)
        search_engine.cache[tok] = result
        if not math.isnan(result["average_score"]):
            token_avg_scores.append(result["average_score"])

    # -------------------------------------------------------
    # Pass 2: Compute global normalization stats (mean/std)
    # -------------------------------------------------------
    if token_avg_scores:
        global_mean = float(np.mean(token_avg_scores))
        global_std = float(np.std(token_avg_scores)) if np.std(token_avg_scores) != 0 else 1.0
    else:
        global_mean, global_std = 0.0, 1.0

    search_engine.global_mean = global_mean
    search_engine.global_std = global_std

    print(f"[INFO] Global normalization: mean={global_mean:.4f}, std={global_std:.4f}")

    # -------------------------------------------------------
    # Pass 3: Normalize cached average scores globally
    # -------------------------------------------------------
    for tok, result in search_engine.cache.items():
        norm_score = (result["average_score"] - global_mean) / global_std
        result["average_score"] = norm_score

    # Save cache to disk
    with open("sentiment_cache.json", "w", encoding="utf-8") as f:
        json.dump(search_engine.cache, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------
    # Pass 4: Predict using globally normalized cached scores
    # -------------------------------------------------------
    preds = []
    for row in tqdm(balanced_df.itertuples(), total=len(balanced_df), desc="Predicting"):
        tokens = row.text.split()
        text_score = np.mean([search_engine.cache.get(tok, {"average_score": 0.0})["average_score"] for tok in tokens])

        pred = "positive" if text_score >= 0.0 else "negative"
        preds.append(pred)

    print("\n" + classification_report(labels, preds, target_names=list(set(labels)), zero_division=0))


if __name__ == "__main__":
    main(args=None) 
