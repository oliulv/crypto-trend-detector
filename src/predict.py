import asyncio
import websockets
import json
import joblib
import os
import requests
import sys
import pandas as pd
from datetime import datetime
from database.classes import Prediction
from feature_engine import LiveFeatureEngine
from database.db import log_prediction
from datetime import datetime, timedelta, timezone
from database.db import SessionLocal


async def spinner_task():
    spinner = ['|', '/', '-', '\\']
    spinner_idx = 0
    # Continually update the spinner until canceled
    while True:
        sys.stdout.write(f"\rWorking... {spinner[spinner_idx]}   ")
        sys.stdout.flush()
        spinner_idx = (spinner_idx + 1) % len(spinner)
        await asyncio.sleep(0.25)  # This controls the spinner update rate


class LiveTradingBot:
    def __init__(self, symbol="PEPEUSDT"):
        self.symbol = symbol
        self.feature_engine = LiveFeatureEngine()
        self.model = self.load_model()
        self.feature_columns = self.get_feature_columns()

    def load_model(self):
        """Load trained model, handling tuple cases (model, calibrator)."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_filename = f'{self.symbol}_pump_predictor.pkl'
        model_path = os.path.join(project_root, 'models', model_filename)

        try:
            model_tuple = joblib.load(model_path)

            # If the loaded object is a tuple, assume the first element is the actual model
            if isinstance(model_tuple, tuple):
                print(f"⚠️ Warning: Model file contains a tuple! Extracting first element...")
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

    async def run(self):
            print("⏳ Loading and connecting to Binance WebSocket stream...")
            spinner_handle = asyncio.create_task(spinner_task())
            outcome_updater = asyncio.create_task(self.update_actual_outcomes())

            uri = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_1m"
            async with websockets.connect(uri) as ws:
                try:
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        candle = data['k']

                        if candle['x']:  # Only process closed candles
                            features = self.feature_engine.process_new_candle(candle)
                            if features is not None:
                                pred_results = self.make_prediction(features)
                                # Pass the entire pred_results dictionary
                                self.handle_prediction(pred_results, candle)

                                # Use the correct fields from pred_results
                                log_prediction(
                                    symbol="PEPE/USDT",
                                    prediction=pred_results['prediction'],
                                    probability=pred_results['probability'],
                                    confidence=pred_results['confidence'],
                                    timestamp=datetime.fromtimestamp(candle['t'] / 1000),
                                    prediction_close=float(candle['c'])  # NEW: Add this line
                                )

                                print("\nWaiting for next candle...")
                except Exception as e:
                    print(f"\nError: {str(e)}")
                finally:
                    spinner_handle.cancel()
                    outcome_updater.cancel()

    def make_prediction(self, features):
        """Convert features to model input and predict"""
        X = pd.DataFrame([features], columns=self.feature_columns)
        X = X.ffill().fillna(0)
        
        proba_all = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]
        
        if prediction == 1:
            proba = proba_all[1]
        else:
            proba = proba_all[0]
        
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

    async def update_actual_outcomes(self):
        """Check every minute for predictions older than 1 hour and update their actual_outcome."""
        while True:
            await asyncio.sleep(60)
            try:
                db = SessionLocal()
                now_utc = datetime.now(timezone.utc)
                one_hour_ago = now_utc - timedelta(hours=1)
                one_hour_ago_naive = one_hour_ago.replace(tzinfo=None)

                predictions = db.query(Prediction).filter(
                    Prediction.timestamp <= one_hour_ago_naive,
                    Prediction.actual_outcome == None
                ).all()

                for pred in predictions:
                    # Fetch ALL candles from T to T+1h
                    start_time = int(pred.timestamp.timestamp() * 1000)
                    end_time = int((pred.timestamp + timedelta(hours=1)).timestamp() * 1000)
                    url = f"https://api.binance.com/api/v3/klines?symbol=PEPEUSDT&interval=1m&startTime={start_time}&endTime={end_time}"

                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        data = response.json()

                        if not data:
                            continue

                        # Extract HIGH prices from all candles in the window
                        high_prices = [float(candle[2]) for candle in data]
                        max_price = max(high_prices) if high_prices else pred.prediction_close

                        # Calculate price change using the MAX price in the window
                        price_change = (max_price - pred.prediction_close) / pred.prediction_close
                        pred.actual_outcome = 1 if price_change >= 0.02 else 0
                        pred.max_hour_close = max_price  # Store max price for reference
                        db.commit()
                        print(f"✅ Updated outcome for prediction {pred.id}")

                    except Exception as e:
                        print(f"🔴 Error for prediction {pred.id}: {str(e)}")

            except Exception as e:
                print(f"❌ Error updating outcomes: {e}")
            finally:
                db.close()


if __name__ == "__main__":
    bot = LiveTradingBot()
    print("🚀 Starting PEPE/USDT Pump Detector...")
    asyncio.run(bot.run())