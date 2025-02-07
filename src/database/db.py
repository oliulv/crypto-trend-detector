import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from database.classes import Prediction
import numpy as np


# Load environment variables from .env file in the project root
load_dotenv()

# Get your PostgreSQL connection URL from the environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create an engine and a session factory
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(symbol, prediction, probability, confidence, timestamp, prediction_close):
    """
    Log a prediction to the database with proper type conversion.
    
    Args:
        symbol (str): Trading pair symbol (e.g., "PEPE/USDT")
        prediction (int/np.integer): The predicted class (0 or 1)
        probability (float/np.floating): Probability of the prediction
        confidence (str): Confidence level ("LOW", "MEDIUM", "HIGH")
        timestamp (datetime): When the prediction was made
    """
    db = SessionLocal()
    try:
        # Convert NumPy types to Python native types
        prediction_value = int(prediction) if isinstance(prediction, (np.integer, np.bool_)) else prediction
        probability_value = float(probability) if isinstance(probability, np.floating) else probability
        prediction_close = float(prediction_close) if isinstance(prediction_close, np.floating) else prediction_close
        
        new_entry = Prediction(
                symbol=symbol,
                prediction=prediction_value,
                probability=probability_value,
                confidence=confidence,
                timestamp=timestamp,
                prediction_close=prediction_close,
                hour_close=None,
                actual_outcome=None
            )
        db.add(new_entry)
        db.commit()
        print(f"✅ Successfully logged prediction to database")
    except Exception as e:
        db.rollback()
        print(f"❌ Error logging prediction: {e}")
        # Add more detailed error information
        print(f"Types - prediction: {type(prediction)}, probability: {type(probability)}")
        print(f"Values - prediction: {prediction}, probability: {probability}")
    finally:
        db.close()