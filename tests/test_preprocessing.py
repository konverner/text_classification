# tests/test_preprocessing.py
import pytest
from text_classification.core.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor_no_stem():
    """Fixture for TextPreprocessor instance without stemming."""
    return TextPreprocessor(stem=False)


@pytest.fixture
def preprocessor_with_stem():
    """Fixture for TextPreprocessor instance with stemming."""
    return TextPreprocessor(stem=True)


def test_preprocess_no_stem(preprocessor_no_stem):
    """Test the preprocess method without stemming."""
    input_text = "This is an example tweet! http://example.com @user"
    expected_output = "example tweet"
    assert preprocessor_no_stem.preprocess(input_text) == expected_output


def test_preprocess_with_stem(preprocessor_with_stem):
    """Test the preprocess method with stemming."""
    input_text = "This is an example tweet! http://example.com @user"
    expected_output = "exampl tweet"
    assert preprocessor_with_stem.preprocess(input_text) == expected_output


def test_preprocess_handles_empty_string(preprocessor_no_stem):
    """Test the preprocess method with an empty string."""
    input_text = ""
    expected_output = ""
    assert preprocessor_no_stem.preprocess(input_text) == expected_output


def test_preprocess_removes_stopwords(preprocessor_no_stem):
    """Test the preprocess method removes stopwords correctly."""
    input_text = "This is an example tweet"
    expected_output = "example tweet"
    assert preprocessor_no_stem.preprocess(input_text) == expected_output


def test_preprocess_removes_special_characters(preprocessor_no_stem):
    """Test the preprocess method removes special characters."""
    input_text = "Example tweet!!! #exciting"
    expected_output = "example tweet"
    assert preprocessor_no_stem.preprocess(input_text) == expected_output
