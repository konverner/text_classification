import pytest
from fastapi.testclient import TestClient

from text_classification.api.main import app
from text_classification.api.schemas import PredictionRequest
from text_classification.core.classifier import TextClassifier

client = TestClient(app)


@pytest.fixture
def mock_classifier():
    classifier = TextClassifier(
        model_path="./tests/units/mocks/mock_model.keras",
        tokenizer_path="./tests/units/mocks/mock_tokenizer.pickle",
        labels=["negative", "positive"],
    )
    return classifier


def test_classify_success(mock_classifier):
    # Arrange
    request_data = PredictionRequest(text="This is a test.")

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 200
    response_data = response.json()
    assert "sentiment" in response_data


def test_classify_empty_text():
    # Arrange
    request_data = PredictionRequest(text="")

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 400
    response_data = response.json()
    assert response_data == {"detail": "Text input is empty"}
