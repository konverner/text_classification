import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from text_classification.api.main import app
from text_classification.api.schemas import PredictionRequest

# Mock data
mock_config = OmegaConf.load("./tests/units/mocks/conf/config.yaml")

mock_prediction = {"label": "positive"}


@pytest.fixture
def client():
    with patch("text_classification.api.routes.OmegaConf.load") as mock_load_config, patch(
        "text_classification.api.routes.instantiate"
    ) as mock_instantiate:
        # Configure the mock objects
        mock_load_config.return_value = mock_config
        mock_text_classifier = MagicMock()
        mock_text_classifier.predict.return_value = [mock_prediction]
        mock_text_preprocessor = MagicMock()
        mock_text_preprocessor.preprocess.return_value = "preprocessed text"

        mock_instantiate.side_effect = (
            lambda conf: mock_text_classifier
            if conf["_target_"] == "your_text_classifier_class"
            else mock_text_preprocessor
        )

        # Create the test client
        client = TestClient(app)
        yield client


def test_classify_success(client):
    # Arrange
    request_data = PredictionRequest(text="This is a test.")
    possible_labels = {"Négatif", "Positif"}

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 200
    response_data = response.json()
    assert "sentiment" in response_data
    assert response_data["sentiment"] in possible_labels


def test_classify_empty_text(client):
    # Arrange
    request_data = PredictionRequest(text="")

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 400
    response_data = response.json()
    assert response_data == {"detail": "Text input is empty"}
