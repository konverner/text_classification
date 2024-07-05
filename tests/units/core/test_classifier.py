import pytest
import numpy as np

from text_classification.core.classifiers import TextClassifier


@pytest.fixture
def mock_classifier():
    classifier = TextClassifier(
        model_path="../tests/units/mocks/mock_model.keras",
        tokenizer_path="../tests/units/mocks/mock_tokenizer.pickle",
        labels=["negative", "positive"],
    )
    return classifier


def test_transform_texts(mock_classifier):
    texts = ["text1", "text2"]
    max_sequence_length = 5
    expected_output = np.array([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6]])  # Example padded sequences
    output = mock_classifier.transform_texts(texts, max_sequence_length)
    np.testing.assert_array_equal(output, expected_output)
