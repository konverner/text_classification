import pytest
import numpy as np

from text_classification.core.classifier import TextClassifier


@pytest.fixture
def mock_classifier():
    classifier = TextClassifier(
        model_path="./tests/units/mocks/mock_model.keras",
        tokenizer_path="./tests/units/mocks/mock_tokenizer.pickle",
        labels=["negative", "positive"],
    )
    return classifier


def test_transform_texts(mock_classifier):
    # Arrange
    texts = ["I love spam", "I love eggs"]

    # Act
    output = mock_classifier.transform_texts(texts, max_sequence_length=3)

    # Assert
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 2


def test_classify_text(mock_classifier):
    # Arrange
    texts = ["I love spam"]
    possible_labels = {"negative", "positive"}

    # Act
    outputs = mock_classifier.predict(texts)

    # Assert
    assert len(outputs) == 1
    for output in outputs:
        assert isinstance(output, dict)
        assert output["label"] in possible_labels
        assert isinstance(output["score"], float)
        assert 0 <= output["score"] <= 1
