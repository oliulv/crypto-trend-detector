import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import websockets
import json
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from database.classes import Prediction
from feature_engine import LiveFeatureEngine
from database.db import SessionLocal
import uvicorn
from log_batcher import PredictionBatcher
import ssl
import certifi
import backoff
from contextlib import contextmanager
import time
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import aiohttp

class LiveTradingBot:
    def __init__(self, symbol="PEPEUSDT"):
        self.symbol = symbol
        self.feature_engine = LiveFeatureEngine()
        self.model = self.load_model()
        self.feature_columns = self.get_feature_columns()
        self.prediction_batcher = PredictionBatcher()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.base_retry_interval = 5  # seconds

    @backoff.on_exception(
        backoff.expo,
        (websockets.ConnectionClosed, websockets.InvalidStatus, ConnectionRefusedError),
        max_tries=5
    )
    async def connect_websocket(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        uri = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_1m"
        
        # Simplified connection parameters
        return await websockets.connect(
            uri,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=60,
            max_size=2**20
        )

    async def run(self):
        while True:  # Outer loop for reconnection
            try:
                print("⏳ Loading and connecting to Binance WebSocket stream...")
                outcome_updater = asyncio.create_task(self.update_actual_outcomes())
                batch_handler = asyncio.create_task(self.handle_batch_predictions())

                async with await self.connect_websocket() as ws:
                    self.reconnect_attempts = 0  # Reset counter on successful connection
                    
                    while True:  # Inner loop for messages
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=65)  # Add timeout
                            data = json.loads(message)
                            candle = data['k']

                            if candle['x']:  # Only process closed candles
                                features = self.feature_engine.process_new_candle(candle)
                                if features is not None:
                                    pred_results = self.make_prediction(features)
                                    self.handle_prediction(pred_results, candle)
                                    
                                    prediction_data = {
                                        "symbol": "PEPE/USDT",
                                        "prediction": pred_results['prediction'],
                                        "probability": pred_results['probability'],
                                        "confidence": pred_results['confidence'],
                                        "timestamp": datetime.fromtimestamp(candle['t'] / 1000),
                                        "prediction_close": float(candle['c']),
                                        "max_hour_close": None,
                                        "actual_outcome": None
                                    }
                                    self.prediction_batcher.add_prediction(prediction_data)
                                    print("\nWaiting for next candle...")

                        except asyncio.TimeoutError:
                            print("No message received within timeout period, sending ping...")
                            pong_waiter = await ws.ping()
                            await asyncio.wait_for(pong_waiter, timeout=10)
                            continue

            except (websockets.ConnectionClosed,
                   websockets.InvalidStatus,
                   ConnectionRefusedError,
                   asyncio.TimeoutError) as e:
                print(f"WebSocket error: {str(e)}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    print("Max reconnection attempts reached. Exiting...")
                    raise

                wait_time = self.base_retry_interval * (2 ** (self.reconnect_attempts - 1))
                print(f"Attempting to reconnect in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise

            finally:
                outcome_updater.cancel()
                batch_handler.cancel()
    
    def load_model(self):
        """Load trained model, handling tuple cases (model, calibrator)."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_filename = f'{self.symbol}_pump_predictor.pkl'
        model_path = os.path.join(project_root, 'models', model_filename)

        try:
            model_tuple = joblib.load(model_path)

            # If the loaded object is a tuple, assume the first element is the actual model
            if isinstance(model_tuple, tuple):
                print(f"Model file contains a tuple! Extracting first element...")
                model = model_tuple[0]  # Extract the trained model
            else:
                model = model_tuple  # If not a tuple, it's already the correct model

            # Ensure the model has `predict_proba`
            if not hasattr(model, "predict_proba"):
                raise AttributeError("❌ The loaded model does NOT support predict_proba()!")

            return model

        except FileNotFoundError:
            raise Exception(f"❌ Model not found at {model_path}. Train first!")

    def get_feature_columns(self):
        """Get feature columns in correct order from training data"""
        # These should match exactly what your model was trained on
        return [
                'open', 'high', 'low', 'close', 'volume', 'num_trades',
                'taker_buy_base', 'taker_buy_quote', '1m_roc', '30m_volatility',
                'volume_zscore_15m', 'buy_pressure', 'hour', 'minute', '1h_volatility',
                '4h_ma_spread', 'volume_spike_15m', 'buy_sell_imbalance', 'hour_sin',
                'hour_cos', 'rsi', 'macd', 'ma_20', 'std_20', 'bollinger_pct_b',
                'ichimoku_conv', 'obv', 'vpt', 'order_imbalance', 'avg_trade_size',
                'large_trade_flag', 'true_range', 'atr', 'stochastic_%k', 'mfi',
                'day_of_week', 'is_weekend', 'asian_session', 'us_session', 'return_skew',
                'price_entropy_1h', '5m_momentum', '15m_momentum', 'spread_ratio',
                'volatility_cluster', 'buy_sell_ratio', 'bid_ask_spread', 'depth_imbalance',
                'fractal_dimension', 'fib_retrace_38', 'fib_retrace_50',
                'order_flow_imbalance', 'rolling_kurtosis', 'lunar_phase'
            ]

    def make_prediction(self, features):
        """Convert features to model input and predict"""
        X = pd.DataFrame([features], columns=self.feature_columns)
        X = X.ffill().fillna(0)
        
        proba_all = self.model.predict_proba(X)[0]
        prediction = int(self.model.predict(X)[0])  # Convert to native int
        
        proba = proba_all[1] if prediction == 1 else proba_all[0]
        proba = float(proba)  # Convert to native float
        
        confidence = 'HIGH' if proba > 0.7 else 'MEDIUM' if proba > 0.5 else 'LOW'
        
        return {
            'prediction': prediction,
            'probability': proba,
            'confidence': confidence
        }

    def handle_prediction(self, pred_results, candle):
        """Handle prediction results"""
        close_time = datetime.fromtimestamp(int(candle['T'])/1000)
        print(f"\n⏰ {close_time} | PEPE/USDT")
        
        if pred_results['prediction'] == 1:
            print(f"🔮 Prediction: PUMP SIGNAL (Confidence: {pred_results['confidence']} [{pred_results['probability']:.20%}])")
        else:
            print(f"🔮 Prediction: No Pump (Confidence: {pred_results['confidence']} [{pred_results['probability']:.20%}])")
        
        print(f"📈 Price: {candle['c']} | Volume: {candle['v']}")

    async def handle_batch_predictions(self):
        """Periodically check and flush batched predictions to database"""
        last_check_time = datetime.now()
        
        while True:
            current_time = datetime.now()
            time_diff = (current_time - last_check_time).seconds
            
            # Only check every minute
            if time_diff >= 60:
                if self.prediction_batcher.should_flush():
                    print(f"🕒 Time since last flush: {(current_time - self.prediction_batcher.last_flush_time).seconds}s")
                    self.prediction_batcher.flush_predictions()
                last_check_time = current_time
            
            await asyncio.sleep(60)  # Sleep for 1 minute before next check

    @contextmanager
    def get_db_session(self, max_retries=5):
        """Context manager for handling database sessions with retry logic"""
        retry_count = 0
        while retry_count < max_retries:
            db = SessionLocal()
            try:
                yield db
                break
            except OperationalError as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise
                print(f"Database connection error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                db.close()
            except Exception:
                db.close()
                raise

    async def update_actual_outcomes(self):
        """Check hourly for predictions that need outcome updates with enhanced error handling"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Wrap the entire processing in a single database session
                with self.get_db_session(max_retries=5) as db:
                    now_utc = datetime.now(timezone.utc)
                    one_hour_ago = now_utc - timedelta(hours=1)
                    one_hour_ago_naive = one_hour_ago.replace(tzinfo=None)

                    try:
                        # Fetch predictions in smaller batches to reduce load
                        batch_size = 100
                        while True:
                            predictions = db.query(Prediction).filter(
                                Prediction.timestamp <= one_hour_ago_naive,
                                Prediction.actual_outcome == None
                            ).limit(batch_size).all()

                            if not predictions:
                                break

                            updates = []
                            for pred in predictions:
                                try:
                                    start_time = int(pred.timestamp.timestamp() * 1000)
                                    end_time = int((pred.timestamp + timedelta(hours=1)).timestamp() * 1000)
                                    
                                    # Add exponential backoff for Binance API calls
                                    for attempt in range(3):
                                        try:
                                            url = f"https://api.binance.com/api/v3/klines?symbol=PEPEUSDT&interval=1m&startTime={start_time}&endTime={end_time}"
                                            async with aiohttp.ClientSession() as session:
                                                async with session.get(url) as response:
                                                    if response.status == 200:
                                                        data = await response.json()
                                                        
                                                        if data:
                                                            high_prices = [float(candle[2]) for candle in data]
                                                            max_price = max(high_prices)
                                                            price_change = (max_price - pred.prediction_close) / pred.prediction_close
                                                            
                                                            pred.actual_outcome = 1 if price_change >= 0.05 else 0
                                                            pred.max_hour_close = max_price
                                                            updates.append(pred)
                                                        break
                                                    else:
                                                        print(f"Binance API returned status {response.status}")
                                                        
                                        except aiohttp.ClientError as e:
                                            if attempt == 2:  # Last attempt
                                                print(f"Failed to fetch data for prediction {pred.id} after 3 attempts: {str(e)}")
                                                break
                                            wait_time = 2 ** attempt
                                            await asyncio.sleep(wait_time)
                                        
                                except Exception as e:
                                    print(f"Error processing prediction {pred.id}: {str(e)}")
                                    continue

                            if updates:
                                try:
                                    # Commit updates in smaller batches
                                    for i in range(0, len(updates), 50):
                                        batch = updates[i:i + 50]
                                        db.bulk_save_objects(batch)
                                        db.commit()
                                    print(f"✅ Updated {len(updates)} prediction outcomes")
                                except SQLAlchemyError as e:
                                    db.rollback()
                                    print(f"❌ Database error while saving updates: {str(e)}")
                                    raise

                            # Small delay between batches to prevent overwhelming the database
                            await asyncio.sleep(1)

                    except SQLAlchemyError as e:
                        print(f"❌ Database query error: {str(e)}")
                        db.rollback()
                        raise

            except Exception as e:
                print(f"❌ Error in update_actual_outcomes: {str(e)}")
                # Wait a bit before retrying the entire process
                await asyncio.sleep(300)  # 5 minutes

async def run_all():
    bot = LiveTradingBot()
    api_task = asyncio.create_task(serve_api())
    await asyncio.gather(bot.run(), api_task)

async def serve_api():
    config = uvicorn.Config("server.api:app", host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    print("🚀 Starting PEPE/USDT Pump Detector and API...")
    asyncio.run(run_all())