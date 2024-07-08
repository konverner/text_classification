from datetime import datetime
from typing import Type

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base: Type = declarative_base()


class SentimentAnalysis(Base):
    """SQLAlchemy model for sentiment analysis results."""

    __tablename__ = "sentiment_analysis"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    document = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    score = Column(Float, nullable=False)
