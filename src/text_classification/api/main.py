import logging

import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from text_classification.api import classify_router

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_app(config_path: str = "src/text_classification/conf/config.yaml") -> FastAPI:
    """
    Create a FastAPI application with the specified configuration.
    Args:
        config_path: The path to the configuration file in yaml format

    Returns:
        FastAPI: The FastAPI application instance.
    """
    config = OmegaConf.load(config_path)

    app = FastAPI(title=config.api.title, description=config.api.description, version=config.api.version)

    # Include the prediction router
    router = classify_router.create_router(config_path)
    app.include_router(router)

    return app


if __name__ == "__main__":
    config_path = "src/text_classification/conf/config.yaml"  # Default path
    config = OmegaConf.load(config_path)
    app = create_app(config_path)
    logger.info("Starting the API server...")
    uvicorn.run(app, host=config.api.host, port=config.api.port)
    logger.info("API server stopped.")
