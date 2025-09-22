"""
Main script to run sentiment classification using different paradigms:
- Supervised: CNN, LSTM, delta-IDF, Morpho-delta
- Semi-supervised: Graph-based random walk labeling
- Unsupervised: Search-based label extraction

Author: Cem Rifki Aydin
Date: 10/01/2020

"""

import argparse
import sys

# Supervised imports
from src.supervised import cnn
from src.supervised import lstm
from src.supervised import delta_idf
from src.supervised import morpho_delta_classifier as morpho  # could be included as supervised if desired

# Other paradigms
from src.unsupervised import search_based_label_extraction as unsupervised
from src.semi_supervised import graph_random_walk_labeling as semi_supervised


def main():
    """
    Main function to parse arguments and route to the appropriate module.
    """
    parser = argparse.ArgumentParser(
        description="Run sentiment classification with supervised, semi-supervised, or unsupervised approaches."
    )

    parser.add_argument(
        "paradigm",
        choices=["supervised", "semi_supervised", "unsupervised"],
        help="Choose learning paradigm"
    )

    parser.add_argument(
        "--method",
        choices=["cnn", "lstm", "delta_idf", "morpho"],
        help="Supervised method (required if paradigm is supervised)"
    )

    parser.add_argument(
        "--lang",
        choices=["english", "turkish"],
        default="english",
        help="Language of the dataset (default: english)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (optional, otherwise each module uses its default dataset)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for neural models (default: 10)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for neural models (default: 16)"
    )

    args = parser.parse_args()

    # --- Validation ---
    if args.paradigm == "supervised" and args.method is None:
        parser.error("--method is required when --paradigm is supervised.")

    if args.method == "morpho" and args.lang != "turkish":
        parser.error("The 'morpho' method is only supported for Turkish (--lang turkish).")
                     

    # Route execution
    if args.paradigm == "supervised":
        if not args.method:
            print("❌ Error: --method must be specified when paradigm is 'supervised'")
            sys.exit(1)
        if args.method == "cnn":
            cnn.main(args)
        elif args.method == "lstm":
            lstm.main(args)
        elif args.method == "delta_idf":
            delta_idf.main(args)
        elif args.method == "morpho":
            morpho.main(args)
        else:
            print("Unknown supervised method:", args.method)
            sys.exit(1)

    elif args.paradigm == "semi_supervised":
        semi_supervised.main(args)

    elif args.paradigm == "unsupervised":
        unsupervised.main(args)

    else:
        print("Unknown paradigm:", args.paradigm)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example commands:

# Supervised CNN (English, default dataset)
python main.py supervised --method cnn --lang english --dataset datasets/english_sentiment_data.csv 

# Supervised LSTM (Turkish, custom dataset, 5 epochs, batch size 16)
python main.py supervised --method lstm --lang turkish --dataset datasets/turkish_sentiment_data.csv --epochs 5 --batch_size 16

# Supervised delta-IDF (Turkish)
python main.py supervised --method delta_idf --lang turkish --dataset datasets/turkish_sentiment_data.csv

# Semi-supervised approach
python main.py --lang english semi_supervised --dataset datasets/english_sentiment_data.csv

# Unsupervised approach
python main.py --lang english unsupervised --dataset datasets/english_sentiment_data.csv

"""