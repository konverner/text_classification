"""
This module provides functionality for text preprocessing, including cleaning, tokenizing,
removing stopwords, and optionally stemming text. It leverages the Natural Language Toolkit (nltk)
for stopword removal and stemming.

Typical usage example:
    To use the TextPreprocessor class, first create an instance of the class, then call the
    `preprocess` method with the text you want to preprocess.

    ```python
    from text_preprocessor import TextPreprocessor

    # Create a TextPreprocessor instance
    preprocessor = TextPreprocessor(stem=True)

    # Text to preprocess
    text = "Check out this link: https://example.com. It's an amazing website! #awesome"

    # Preprocess the text
    processed_text = preprocessor.preprocess(text)

    print(processed_text)
    # Output: check amaz websit
"""

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download the stopwords from NLTK
nltk.download("stopwords")


class TextPreprocessor:
    """
    A class used to preprocess text by cleaning, tokenizing, removing stopwords, and optionally stemming.

    Attributes:
        stop_words: A list of stopwords to be removed from the text.
        stemmer: A stemmer used to reduce words to their root form.
        text_cleaning_re: A regular expression pattern for cleaning text.
        stem: A flag to indicate whether to apply stemming.
    """

    def __init__(self, stem: bool = False, language: str = "english"):
        """
        Initializes the TextPreprocessor with the option to apply stemming.

        Args:
            stem (bool): If True, apply stemming to the tokens. Default is False.
        """
        self.stop_words: List[str] = stopwords.words(language)
        self.stemmer: SnowballStemmer = SnowballStemmer(language)
        self.text_cleaning_re: str = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        self.stem: bool = stem

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the input text by removing unwanted characters, converting to lowercase,
        removing stopwords, and optionally applying stemming.

        Args:
            text: The input text to preprocess.

        Returns:
            The preprocessed text.
        """
        text = re.sub(self.text_cleaning_re, " ", text.lower()).strip()
        tokens: List[str] = []
        for token in text.split():
            if token not in self.stop_words:
                if self.stem:
                    tokens.append(self.stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)
