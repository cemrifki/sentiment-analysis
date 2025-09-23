# Sentiment analysis in Turkish: Supervised, semi-supervised, and unsupervised techniques

This repo contains the source code of the paper --

[Sentiment analysis in Turkish: Supervised, semi-supervised, and unsupervised techniques](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/sentiment-analysis-in-turkish-supervised-semisupervised-and-unsupervised-techniques/3E5CAB8E6A2B8877135F63485536C8F9). Cem Rıfkı Aydın, Tunga Güngör. Natural Language Engineering, **Cambridge University Press**.

In this study, sentiment analysis is performed in Turkish and English. To achieve this, unsupervised, semi-supervised, and supervised approaches have been developed and applied. The supervised methods include LSTM, CNN, and delta-IDF techniques. For Turkish, a morphological analysis is also incorporated, further improving performance. The methods are evaluated on datasets from various genres, making the approach cross-domain and easily adaptable to other languages. The code that was originally written in Java has now been rewritten in Python.

## Requirements

- numpy
- tensorflow
- tqdm
- scikit-learn
- scipy
- pandas
- nltk
- jpype1

In addition to Python packages, the Zemberek tool, originally written in Java, was also used for high-quality morphological analysis of Turkish texts. To enable this, the jpype1 library was utilized.

## Setup Instructions
1. Clone the repository.
2. Create Python virtual environment (e.g., `python -m venv my_venv`).
3. Activate it: `source my_venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`.

I leveraged `Python 3.9` when running the code.

## Project Structure

``` bash
sentiment-analysis/
├── datasets/                         # Raw and processed datasets
│   ├── turkish/                      # Turkish datasets
│   └── english/                      # English datasets
│
├── resources/                        # External resources (Zemberek used for morphological parsing)
│   └── zemberek-full.jar
│
├── src/                              # Core source code
│   ├── supervised/                   # Supervised learning models
│   │   ├── cnn.py
│   │   ├── delta_idf.py
│   │   ├── lstm.py
│   │   └── morpho_delta_classifier.py
│   │
│   ├── semi_supervised/              # Semi-supervised methods
│   │   └── graph_random_walk_labeling.py
│   │
│   ├── unsupervised/                 # Unsupervised methods
│   │   └── search_based_label_extraction.py
│   │
│   ├── utils/                        # Helper functions
│   │   ├── seeds.py
│   │   └── utils.py
│   │
│   └── __init__.py
│
├── LICENSE
├── README.md
├── requirements.txt
└── main.py                           # Entrypoint (train/eval script)
```

## Usage

Run the below exemplary command to run the CNN method for English.

```bash
python main.py supervised --method cnn --lang english --dataset datasets/english/english_sentiment_data.csv 
```

If you only want to run the LSTM approach for Turkish, you can utilize the below exemplary command:

```bash
python main.py supervised --method lstm --lang turkish --dataset datasets/turkish/turkish_sentiment_data.csv --epochs 5 --batch_size 32

```
## Citation

If you find this code useful, please cite the following in your work:

```
@article{Aydın_Güngör_2021,
  author={Aydin, Cem Rifki and Güngör, Tunga},
  journal={Natural Language Engineering}, 
  title={Sentiment analysis in Turkish: Supervised, semi-supervised, and unsupervised techniques}, 
  year={2021},
  volume={27},
  number={4},
  pages={455–483},
  keywords={Sentiment analysis;Opinion mining;Machine learning;Text classification;Morphological analysis},
  DOI={10.1017/S1351324920000200}
}
```

## Credits
The code was written by Cem Rifki Aydin