import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add root directory to path

from collections import deque
from datetime import datetime
from db.classes import Prediction
from db.db import SessionLocal
from typing import Dict
import threading
import time
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from contextlib import contextmanager

class PredictionBatcher:
    def __init__(self, batch_size: int = 60, flush_interval: int = 3600, max_retries: int = 5):
        self.predictions_queue = deque()
        self.batch_size = batch_size
        self.flush_interval = flush_interval  # 3600 seconds = 1 hour
        self.last_flush_time = datetime.now()
        self.lock = threading.Lock()
        self.max_retries = max_retries
        self._force_flush = False  # New flag for forced flushes

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
        # Check time since last flush
        time_since_last_flush = (datetime.now() - self.last_flush_time).seconds
        return (len(self.predictions_queue) >= self.batch_size or 
                time_since_last_flush >= self.flush_interval or
                self._force_flush)
    
    # In case we need it for manual use later.
    def force_flush(self):
        """Force a flush on next check"""
        self._force_flush = True
        
    def flush_predictions(self) -> None:
        with self.lock:
            if not self.predictions_queue:
                self._force_flush = False
                return

            predictions_to_process = list(self.predictions_queue)
            self.predictions_queue.clear()
            
            # Process in smaller sub-batches
            sub_batch_size = 10
            successful_predictions = []
            
            print(f"🔄 Processing {len(predictions_to_process)} predictions in smaller batches...")
            
            for i in range(0, len(predictions_to_process), sub_batch_size):
                sub_batch = predictions_to_process[i:i+sub_batch_size]
                prediction_objects = [Prediction(**pred_data) for pred_data in sub_batch]
                
                success = False
                retry_count = 0
                max_retries = 3
                
                while not success and retry_count < max_retries:
                    try:
                        with self.get_db_session() as db:
                            try:
                                db.bulk_save_objects(prediction_objects)
                                db.commit()
                                successful_predictions.extend(sub_batch)
                                print(f"✅ Sub-batch {i//sub_batch_size + 1} logged {len(prediction_objects)} predictions successfully")
                                success = True
                            except SQLAlchemyError as e:
                                db.rollback()
                                retry_count += 1
                                if retry_count == max_retries:
                                    print(f"❌ Failed to save sub-batch after {max_retries} attempts: {str(e)}")
                                    # Put failed predictions back in queue
                                    with self.lock:
                                        for pred in sub_batch:
                                            self.predictions_queue.append(pred)
                                else:
                                    wait_time = 2 ** retry_count
                                    print(f"Retrying sub-batch in {wait_time}s (attempt {retry_count}/{max_retries})")
                                    time.sleep(wait_time)
                    except Exception as e:
                        print(f"❌ Connection error: {str(e)}")
                        retry_count += 1
                        if retry_count == max_retries:
                            # Put failed predictions back in queue
                            with self.lock:
                                for pred in sub_batch:
                                    self.predictions_queue.append(pred)
                            break
                        time.sleep(2 ** retry_count)
            
            # Update last flush time if any predictions were successful
            if successful_predictions:
                self.last_flush_time = datetime.now()
                print(f"✅ Total logged: {len(successful_predictions)}/{len(predictions_to_process)} predictions")
            else:
                print("⚠️ No predictions were successfully logged")
                
            self._force_flush = False