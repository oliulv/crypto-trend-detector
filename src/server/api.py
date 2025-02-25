from fastapi import FastAPI
from db.db import SessionLocal
from db.classes import Prediction
from typing import List

app = FastAPI()

# Endpoint for getting predictions made and stored in the database so far
@app.get("/predictions", response_model=List[dict])
async def get_predictions(limit: int = 100):
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
        return [
            {
                "timestamp": pred.timestamp,
                "prediction": pred.prediction,
                "probability": pred.probability,
                "price": pred.prediction_close
            }
            for pred in predictions
        ]
    finally:
        db.close()

# Endpoint for seeing the status of the bot
@app.get("/status")
async def get_status():
    return {"status": "active", "message": "Bot is running"}