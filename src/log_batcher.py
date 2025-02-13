from collections import deque
from datetime import datetime
from database.classes import Prediction
from database.db import SessionLocal
from typing import Dict
import threading
import time
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from contextlib import contextmanager

class PredictionBatcher:
    def __init__(self, batch_size: int = 60, flush_interval: int = 3600, max_retries: int = 5):
        self.predictions_queue = deque()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush_time = datetime.now()
        self.lock = threading.Lock()
        self.max_retries = max_retries

    @contextmanager
    def get_db_session(self):
        """Context manager for handling database sessions with retry logic"""
        retry_count = 0
        while retry_count < self.max_retries:
            db = SessionLocal()
            try:
                yield db
                break
            except OperationalError as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise
                print(f"Database connection error (attempt {retry_count}/{self.max_retries}): {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                db.close()
            except Exception:
                db.close()
                raise

    def add_prediction(self, prediction_data: Dict):
        with self.lock:
            self.predictions_queue.append(prediction_data)
    
    def should_flush(self) -> bool:
        return (len(self.predictions_queue) >= self.batch_size or 
                (datetime.now() - self.last_flush_time).seconds >= self.flush_interval)
    
    def flush_predictions(self) -> None:
        with self.lock:
            if not self.predictions_queue:
                return

            predictions = []
            # Create prediction objects outside the database transaction
            while self.predictions_queue:
                pred_data = self.predictions_queue.popleft()
                predictions.append(Prediction(**pred_data))

            try:
                with self.get_db_session() as db:
                    try:
                        db.bulk_save_objects(predictions)
                        db.commit()
                        print(f"✅ Batch logged {len(predictions)} predictions to database")
                        self.last_flush_time = datetime.now()
                    except SQLAlchemyError as e:
                        db.rollback()
                        raise e
            except Exception as e:
                print(f"❌ Error batch logging predictions: {e}")
                # Put failed predictions back in queue
                with self.lock:
                    for pred in predictions:
                        pred_dict = pred.__dict__
                        pred_dict.pop('_sa_instance_state', None)
                        self.predictions_queue.appendleft(pred_dict)