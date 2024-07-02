"""
This module contains tools for text preprocessing, including cleaning,
tokenizing, removing stopwords, and stemming. The main class provided is
`TextPreprocessor`, which offers methods for preprocessing text data.

Usage Example:
    preprocessor = TextPreprocessor(stem=True)
    clean_text = preprocessor.preprocess("Your raw tweet text here")
"""

import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download the stopwords from NLTK
nltk.download('stopwords')

class TextPreprocessor:
    """
    A class used to preprocess text by cleaning, tokenizing, removing stopwords, and optionally stemming.

    Attributes:
        stop_words (List[str]): A list of stopwords to be removed from the text.
        stemmer (SnowballStemmer): A stemmer used to reduce words to their root form.
        text_cleaning_re (str): A regular expression pattern for cleaning text.
        stem (bool): A flag to indicate whether to apply stemming.
    """

    def __init__(self, stem: bool = False):
        """
        Initializes the TextPreprocessor with the option to apply stemming.

        Args:
            stem (bool): If True, apply stemming to the tokens. Default is False.
        """
        self.stop_words: List[str] = stopwords.words('english')
        self.stemmer: SnowballStemmer = SnowballStemmer('english')
        self.text_cleaning_re: str = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        self.stem: bool = stem

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the input text by removing unwanted characters, converting to lowercase,
        removing stopwords, and optionally applying stemming.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(self.text_cleaning_re, ' ', text.lower()).strip()
        tokens: List[str] = []
        for token in text.split():
            if token not in self.stop_words:
                if self.stem:
                    tokens.append(self.stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)
