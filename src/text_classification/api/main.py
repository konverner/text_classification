import logging

import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from text_classification.api.routes import router

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize Hydra and load configuration
config = OmegaConf.load("src/text_classification/conf/config.yaml")

app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version
)

# Include the prediction router
app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting the API server...")
    uvicorn.run(app, host=config.api.host, port=config.api.port)
    logger.info("API server stopped.")
