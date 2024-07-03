import logging

from fastapi import APIRouter, HTTPException
from omegaconf import OmegaConf
from hydra.utils import instantiate

from text_classification.api.schemas import PredictionRequest, PredictionResponse

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize Hydra and load configuration
config = OmegaConf.load("src/text_classification/conf/config.yaml")

router = APIRouter()

logger.info("Loading pre-trained sentiment analysis model...")
text_classifier = instantiate(config.text_classifier)
text_preprocessor = instantiate(config.text_preprocessor)
logger.info("Model loaded successfully.")

@router.post("/classify", response_model=PredictionResponse)
async def classify(request: PredictionRequest):
    text = request.text

    if not text:
        logger.warning("Received empty text input.")
        raise HTTPException(status_code=400, detail="Text input is empty")

    logger.info(f"Received text: {text}")

    # Preprocess the text
    preprocessed_text = text_preprocessor.preprocess(text)

    logger.info(f"Preprocessed text: {text}")

    # Perform prediction
    result = text_classifier.predict([preprocessed_text])[0]
    logger.info(f"Prediction result: {result}")

    return PredictionResponse(sentiment=result["label"])
