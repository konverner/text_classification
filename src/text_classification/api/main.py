import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from sqlalchemy.ext.asyncio import create_async_engine

from text_classification.api import classify_router
from text_classification.db.models import Base

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


async def create_tables():
    """Create the database tables."""

    config = OmegaConf.load("src/text_classification/conf/config.yaml")
    engine = create_async_engine(config.database.url, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_app(config_path: str = "src/text_classification/conf/config.yaml") -> FastAPI:
    """Create a FastAPI application."""

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
    asyncio.run(create_tables())
    logger.info("Starting the API server...")
    uvicorn.run(app, port=config.api.port)
    logger.info("API server stopped.")
