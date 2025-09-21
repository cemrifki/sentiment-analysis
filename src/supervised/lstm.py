"""
Long Short-Term Memory (LSTM) Network for Text Classification using TensorFlow/Keras.

Author: Cem Rifki Aydin
Date: 18/04/2020

"""

import pandas as pd
import numpy as np

import nltk

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')

from src.utils.utils import preprocess_text


def main(args):

    lang=args.lang
    dataset_path=args.dataset
    epochs=args.epochs
    batch_size=args.batch_size

    df = pd.read_csv(dataset_path)
    df['Clean_Text'] = df['text'].apply(lambda x: preprocess_text(x, lang=lang))

    le = LabelEncoder()
    df['Sentiment_Enc'] = le.fit_transform(df['sentiment'])
    num_classes = len(le.classes_)

    MAX_NUM_WORDS = 1000
    MAX_SEQUENCE_LENGTH = 25
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['Clean_Text'])

    X = tokenizer.texts_to_sequences(df['Clean_Text'])
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    y = df['Sentiment_Enc'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.002),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))


if __name__ == "__main__":
    main()
