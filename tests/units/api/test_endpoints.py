import pytest
from fastapi.testclient import TestClient

from text_classification.api.main import create_app
from text_classification.api.schemas import PredictionRequest


@pytest.fixture
def client():
    config_path = "./tests/units/mocks/conf/config.yaml"
    app = create_app(config_path)
    client = TestClient(app)
    return client


def test_classify_success(client):
    # Arrange
    request_data = PredictionRequest(text="This is a test.")

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 200
    response_data = response.json()
    assert "sentiment" in response_data


def test_classify_empty_text(client):
    # Arrange
    request_data = PredictionRequest(text="")

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 400
    response_data = response.json()
    assert response_data == {"detail": "Text input is empty"}


def test_classify_too_long_text(client):
    # Arrange
    request_data = PredictionRequest(text="a" * 2025)

    # Act
    response = client.post("/classify", json=request_data.dict())

    # Assert
    assert response.status_code == 400
    response_data = response.json()
    assert response_data == {"detail": "Text input is too long, max length is 2024"}
