import logging

from fastapi import APIRouter, HTTPException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from text_classification.api.schemas import ClassificationResult, PredictionRequest, PredictionResponse

# Constants
DEFAULT_CONFIG_PATH = "src/text_classification/conf/config.yaml"
MAX_LENGTH_TEXT = 2024

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_router(config_path: str = DEFAULT_CONFIG_PATH) -> APIRouter:
    """
    Create an API router for the text classification service.

    Args:
        config_path: The path to the configuration file in yaml format

    Returns:
        An APIRouter instance with the classification endpoint.

    Raises:
        RuntimeError: If an error occurs during model or preprocessor loading.
    """
    # Initialize Hydra and load configuration
    config = OmegaConf.load(config_path)

    router = APIRouter()

    logger.info("Loading model...")
    try:
        text_classifier = instantiate(config.text_classifier)
        text_preprocessor = instantiate(config.text_preprocessor)
    except Exception as e:
        logger.error(f"Error instantiating model or preprocessor: {e}")
        raise RuntimeError("Failed to load model or preprocessor") from e
    logger.info("Model loaded successfully.")

    @router.post("/classify", response_model=PredictionResponse)
    async def classify(request: PredictionRequest):
        """Classifies a given text.

        Args:
            request: The request containing the text to be classified.

        Returns:
            The response containing the sentiment label.

        Raises:
            HTTPException: If the text input is empty or an error occurs during preprocessing or prediction.
        """
        texts = request.texts
        for text in texts:
            if not text:
                logger.warning("Received empty text input.")
                raise HTTPException(status_code=400, detail="One of text inputs is empty")
            if len(text) > MAX_LENGTH_TEXT:
                logger.error(f"Received too long text: {len(text)}")
                raise HTTPException(
                    status_code=400, detail=f"One of text inputs is too long, max length is {MAX_LENGTH_TEXT}"
                )

        logger.info(f"Received text: {text}")

        # Preprocess the text
        try:
            preprocessed_texts = text_preprocessor.preprocess(texts)
        except Exception as e:
            logger.error(f"Error during text preprocessing: {e}")
            raise HTTPException(status_code=500, detail="Error during text preprocessing") from e

        # Perform prediction
        try:
            results = text_classifier.predict(preprocessed_texts)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail="Error during prediction") from e

        classification_results = []
        for result in results:
            classification_results.append(ClassificationResult(text=text, label=result["label"], score=result["score"]))
        return PredictionResponse(outputs=classification_results)

    return router
