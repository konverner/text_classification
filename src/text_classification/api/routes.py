import logging

from fastapi import APIRouter, HTTPException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from text_classification.api.schemas import PredictionRequest, PredictionResponse

# Constants
CONFIG_PATH = "src/text_classification/conf/config.yaml"
MAX_LENGTH_TEXT = 2024

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize Hydra and load configuration
config = OmegaConf.load("src/text_classification/conf/config.yaml")

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
    text = request.text
    input_text_length = len(text)

    if not text:
        logger.warning("Received empty text input.")
        raise HTTPException(status_code=400, detail="Text input is empty")

    if input_text_length > MAX_LENGTH_TEXT:
        logger.error(f"Received too long text: {input_text_length}")
        raise HTTPException(status_code=400, detail=f"Text input is too long, max length is {MAX_LENGTH_TEXT}")

    logger.info(f"Received text: {text}")

    # Preprocess the text
    try:
        preprocessed_text = text_preprocessor.preprocess(text)
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Error during text preprocessing") from e

    logger.info(f"Preprocessed text: {text}")

    # Perform prediction
    try:
        result = text_classifier.predict([preprocessed_text])[0]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction") from e

    return PredictionResponse(sentiment=result["label"])
