from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime


Base = declarative_base()


class Prediction(Base):
    """
    Database model for storing trading predictions.
    Maps directly to predictions table in the database.
    """
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    symbol = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)  
    probability = Column(Float, nullable=False)   
    confidence = Column(String, nullable=False)
    actual_outcome = Column(Integer, nullable=True)
