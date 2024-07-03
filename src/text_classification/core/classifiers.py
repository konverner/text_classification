import pickle
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextClassifier:
    """
    A class to handle text classification using TensorFlow and pre-trained model.

    Attributes:
        tokenizer (Tokenizer): Tokenizer instance for text processing.
        model (tf.keras.Model): The pre-trained Keras model for text classification.
    """

    def __init__(self, model_path: str, tokenizer_path: str, labels: List[str]) -> None:
        """
        Initialize the TextClassifier with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model file.
        """
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.labels = labels

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> Tokenizer:
        """
        Load the tokenizer.

        Args:
            tokenizer_path str: Path to the tokenizer.
        Returns:

        """
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """
        Load a model from a file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            tf.keras.Model: Loaded Keras model.
        """
        return tf.keras.models.load_model(model_path)

    def transform_texts(self, texts: List[str], max_sequence_length: int) -> np.ndarray:
        """
        Transform texts to padded sequences.

        Args:
            texts (List[str]): List of texts to transform.
            max_sequence_length (int): Maximum length of sequences.

        Returns:
            np.ndarray: Padded sequences.
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
            List[dict]: List of prediction results.
        """
        max_sequence_length = self.model.input_shape[1]
        transformed_texts = self.transform_texts(texts, max_sequence_length)
        scores = self.model.predict(transformed_texts)
        results = []
        for score in scores:
            # 1 if score >= 0.5 else 0
            label_index = int(score >= 0.5)
            label = self.labels[label_index]
            results.append({'label': label, 'score': float(score)})
        return results
