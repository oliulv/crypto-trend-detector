from collections import deque
from datetime import datetime
from database.classes import Prediction
from database.db import SessionLocal
from typing import Dict
import threading


class PredictionBatcher:
    def __init__(self, batch_size: int = 60, flush_interval: int = 3600):
        self.predictions_queue = deque()
        self.batch_size = batch_size  # Number of predictions to batch before writing
        self.flush_interval = flush_interval  # Seconds between forced flushes
        self.last_flush_time = datetime.now()
        self.lock = threading.Lock()
        
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
                
            db = SessionLocal()
            try:
                # Bulk insert all predictions in the queue
                predictions = []
                while self.predictions_queue:
                    pred_data = self.predictions_queue.popleft()
                    predictions.append(Prediction(**pred_data))
                
                db.bulk_save_objects(predictions)
                db.commit()
                print(f"✅ Batch logged {len(predictions)} predictions to database")
                self.last_flush_time = datetime.now()
                
            except Exception as e:
                db.rollback()
                print(f"❌ Error batch logging predictions: {e}")
                # Put failed predictions back in queue
                with self.lock:
                    for pred in predictions:
                        pred_dict = pred.__dict__
                        pred_dict.pop('_sa_instance_state', None)
                        self.predictions_queue.appendleft(pred_dict)
            finally:
                db.close()