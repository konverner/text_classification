from omegaconf import OmegaConf
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

config = OmegaConf.load("src/text_classification/conf/config.yaml")
DATABASE_URL = config.database.url

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)


async def get_db():
    """Get a database session."""
    async with SessionLocal() as session:
        yield session
