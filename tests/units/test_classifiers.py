import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from text_classification.core.classifiers import TextClassifier

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.input_shape = (None, 100)  # Example input shape
    model.predict.return_value = np.array([[0.8], [0.3]])  # Example prediction
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3], [4, 5, 6]]  # Example tokenized sequences
    return tokenizer

@pytest.fixture
def labels():
    return ['negative', 'positive']

@patch('text_classifier.pickle.load')
@patch('builtins.open', new_callable=mock_open)
def test_load_tokenizer(mock_open, mock_pickle_load, mock_tokenizer):
    mock_pickle_load.return_value = mock_tokenizer
    tokenizer_path = 'dummy_tokenizer_path.pkl'
    tokenizer = TextClassifier.load_tokenizer(tokenizer_path)
    mock_open.assert_called_once_with(tokenizer_path, 'rb')
    assert tokenizer == mock_tokenizer

@patch('tensorflow.keras.models.load_model')
def test_load_model(mock_load_model, mock_model):
    model_path = 'dummy_model_path.h5'
    mock_load_model.return_value = mock_model
    model = TextClassifier.load_model(model_path)
    mock_load_model.assert_called_once_with(model_path)
    assert model == mock_model

@patch.object(TextClassifier, 'load_model')
@patch.object(TextClassifier, 'load_tokenizer')
def test_init(mock_load_tokenizer, mock_load_model, mock_model, mock_tokenizer, labels):
    mock_load_model.return_value = mock_model
    mock_load_tokenizer.return_value = mock_tokenizer
    model_path = 'dummy_model_path.h5'
    tokenizer_path = 'dummy_tokenizer_path.pkl'
    classifier = TextClassifier(model_path, tokenizer_path, labels)
    assert classifier.model == mock_model
    assert classifier.tokenizer == mock_tokenizer
    assert classifier.labels == labels

def test_transform_texts(mock_tokenizer, labels):
    classifier = TextClassifier('dummy_model_path.h5', 'dummy_tokenizer_path.pkl', labels)
    classifier.tokenizer = mock_tokenizer
    texts = ['text1', 'text2']
    max_sequence_length = 5
    expected_output = np.array([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6]])  # Example padded sequences
    output = classifier.transform_texts(texts, max_sequence_length)
    np.testing.assert_array_equal(output, expected_output)

@patch.object(TextClassifier, 'transform_texts')
def test_predict(mock_transform_texts, mock_model, labels):
    classifier = TextClassifier('dummy_model_path.h5', 'dummy_tokenizer_path.pkl', labels)
    classifier.model = mock_model
    texts = ['text1', 'text2']
    mock_transform_texts.return_value = np.array([[1, 2, 3], [4, 5, 6]])  # Example transformed texts
    results = classifier.predict(texts)
    expected_results = [{'label': 'positive', 'score': 0.8}, {'label': 'negative', 'score': 0.3}]
    assert results == expected_results
