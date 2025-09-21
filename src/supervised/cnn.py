""" 
Convolutional Neural Network (CNN) for Text Classification using TensorFlow/Keras.

Author: Cem Rifki Aydin
Date: 15/04/2020

"""

# Data manipulation
import numpy as np
import pandas as pd

# NLP
import nltk
from src.utils.utils import preprocess_text  # custom preprocessing

# Sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow / Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

# Setup (download resources, constants, etc.)
nltk.download('punkt')


def main(args):

    dataset_path = args.dataset
    lang = args.lang
    dataset_path = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    
    print(f"[CNN] Running with lang={lang}, dataset={dataset_path}, epochs={epochs}, batch_size={batch_size}")
    # load dataset depending on lang or dataset_path

    df = pd.read_csv(dataset_path)
    df = df[['text', 'sentiment']].dropna()

    df['Clean_Text'] = df['text'].apply(lambda x: preprocess_text(x, lang=lang))

    le = LabelEncoder()
    df['Sentiment_Enc'] = le.fit_transform(df['sentiment'])

    MAX_NUM_WORDS = 10_000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Clean_Text'])
    sequences = tokenizer.texts_to_sequences(df['Clean_Text'])
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = df['Sentiment_Enc'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    num_classes = len(le.classes_)
    model = Sequential([
        Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    pred_sentiments = le.inverse_transform(pred_labels)


if __name__ == "__main__":
    main(args=None)
