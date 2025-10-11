"""
Search-Based Unsupervised Label Extraction using Yandex Search Engine.

Author: Cem Rifki Aydin
Date: 12/01/2020

"""

# ------------------------------
# Standard library imports
# ------------------------------
import json
import math
import re
import threading
import time
from typing import Dict, List

# ------------------------------
# Third-party imports
# ------------------------------
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
    TURKISH_CHARS = "ğüşıöçĞÜŞİÖÇîâûÎÂÛ"
    EQUIV = [
        "%C4%9F", "%C3%BC", "%C5%9F", "%C4%B1", "%C3%B6", "%C3%A7",
        "%C4%9E", "%C3%9C", "%C5%9E", "%C4%B0", "%C3%96", "%C3%87",
        "%C3%AE", "%C3%A2", "%C3%BB", "%C3%8E", "%C3%82", "%C3%9B"
    ]

    def __init__(self, lang: str, near: int = 4, time_conn: int = 5):
        self.constants = Constants(lang=lang)
        self.lang = lang
        self.near = near  # <-- configurable NEAR window size
        self.time_conn = time_conn
        
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

    def query_result(self, query_word: str, related_word: str) -> str:
        """Query Yandex and return the number of results as string."""
        near = f"%2F{self.near}"   # dynamically inject near value
        space = "%20"
        all_query = f'{query_word}{space}{near}{space}{related_word}'
        for c, e in zip(self.TURKISH_CHARS, self.EQUIV):
            all_query = all_query.replace(c, e)
        all_query = all_query.replace('"', '%22')
        url = f"https://www.yandex.com.tr/search/?text={all_query}&lr=11508"
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
        target = f'"{target_word}"'
        pos_word = f'"{pair[0]}"'
        neg_word = f'"{pair[1]}"'

        pos_val = self.query_result(target, pos_word)
        neg_val = self.query_result(target, neg_word)

        score = math.log10(
            float(self.str_to_db_yan(pos_val)) / float(self.str_to_db_yan(neg_val))
        )

        result_obj = {
            "target_word": target_word,
            "positive_word": pair[0],
            "positive_count": pos_val,
            "negative_word": pair[1],
            "negative_count": neg_val,
            "score": score
        }

        results_list.append(result_obj)
        # if Constants.PRINT_RESULTS:
        #     print(json.dumps(result_obj, ensure_ascii=False, indent=2))

    def compute_sentiment(self, target_word: str) -> Dict:
        """
        Compute sentiment of the target word using all WORD_PAIRS.
        Returns JSON with pair scores and average score.
        """
        threads = []
        results_list: List[Dict] = []

        # Start a thread for each word pair
        for pair in self.constants.WORD_PAIRS:
            t = threading.Thread(target=self.calc_pair_score, args=(target_word, pair, results_list))
            threads.append(t)
            t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Compute overall average
        total_score = sum(result["score"] for result in results_list)
        average_score = total_score / len(self.constants.WORD_PAIRS)

        overall_result = {
            "target_word": target_word,
            "pair_results": results_list,
            "average_score": average_score
        }

        return overall_result

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
    """Main function to run the search-based unsupervised label extraction."""

    dataset_path=args.dataset
    df = pd.read_csv(dataset_path)  # must have columns: text, sentiment

    df["sentiment"] = (df["sentiment"].str.lower().
                     map(LABEL_MAP)
                     )

    # In fact, a balanced dataset is not needed here for the unsupervised approach, but for testing purposes    
    balanced_df = make_balanced(df, label_col="sentiment", random_state=42)

    # If you just want the text column as a list
    corpus = balanced_df["text"].tolist()
    labels = balanced_df["sentiment"].tolist()


    corpus = filter_by_frequency([tokenize(text, args.lang) for text in corpus])
    corpus = [" ".join(text) for text in corpus]

    search_engine = SearchEngine(args.lang, args.near if args.near else 4)

    le = LabelEncoder()
    balanced_df['Sentiment_Enc'] = le.fit_transform(balanced_df['sentiment'])
    num_classes = len(le.classes_)

    preds = []
    for row in tqdm(balanced_df.itertuples(), total=len(balanced_df)):
        tokens = row.text.split()
        tok_scores = [search_engine.compute_sentiment(tok)["average_score"] for tok in tokens]
        pred = sum(tok_scores)
        pred = "positive" if pred > 0 else "negative"
        preds.append(pred)

    print(classification_report(labels, preds, target_names=le.classes_, zero_division=0))


if __name__ == "__main__":
    main(args=None) 
