"""
This module provides the `TextClassifier` class for text classification using TensorFlow and a pre-trained model.

Typical usage example:

    model_path = 'path/to/model.keras'
    tokenizer_path = 'path/to/tokenizer.pickle'
    labels = ['negative', 'positive']

    classifier = TextClassifier(model_path, tokenizer_path, labels)

    texts = ["This is a great movie!", "I did not like this film."]
    predictions = classifier.predict(texts)

    for prediction in predictions:
        print(f"Label: {prediction['label']}, Score: {prediction['score']}")
"""

import pickle
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class TextClassifier:
    """
    A class to handle text classification using TensorFlow and pre-trained model.

    Attributes:
        tokenizer (Tokenizer): Tokenizer instance for text processing.
        model (tf.keras.Model): The pre-trained Keras model for text classification.
    """

    def __init__(self, model_path: str, tokenizer_path: str, labels: List[str]):
        """
        Initialize the TextClassifier with a pre-trained model.

        Args:
            model_path: Path to the pre-trained model file.
            tokenizer_path: Path to the tokenizer pickle file.
        """
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.labels = labels

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> Tokenizer:
        """
        Load the tokenizer.

        Args:
            tokenizer_path: Path to the tokenize.

        Returns:
            Loaded keras Tokenizer instance.
        """
        with open(tokenizer_path, "rb") as handle:  # noqa: S301
            tokenizer = pickle.load(handle)  # noqa: S301
        return tokenizer

    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """
        Load a model from a file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            Loaded Keras model.
        """
        return tf.keras.models.load_model(model_path)

    def transform_texts(self, texts: List[str], max_sequence_length: int) -> np.ndarray:
        """
        Transform texts to padded sequences.

        Args:
            texts: List of texts to transform.
            max_sequence_length: Maximum length of sequences.

        Returns:
            Padded sequences.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        return padded_sequences

    def predict(self, texts: List[str]) -> List[dict]:
        """
        Predict the sentiment of the input texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List of prediction results where each result is a dictionary with label and corresponding score
            For example: [{'label': 'positive', 'score': 0.42}]
        """
        max_sequence_length = self.model.input_shape[1]
        transformed_texts = self.transform_texts(texts, max_sequence_length)
        scores = self.model.predict(transformed_texts)
        results = []
        for score in scores:
            # 1 if score >= 0.5 else 0
            label_index = int(score >= 0.5)
            label = self.labels[label_index]
            results.append({"label": label, "score": float(score)})
        return results
