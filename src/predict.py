import asyncio
import websockets
import json
import joblib
import os
import pandas as pd
from datetime import datetime
from feature_engine import LiveFeatureEngine
import sys


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
        model_path = os.path.join(project_root, 'model', model_filename)

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
            'hour_cos', 'rsi', 'macd', 'bollinger_pct_b', 'ichimoku_conv', 'obv',
            'vpt', 'order_imbalance', 'avg_trade_size', 'large_trade_flag', 'true_range',
            'atr', 'stochastic_%k', 'mfi', 'day_of_week', 'is_weekend', 'asian_session',
            'us_session', 'return_skew', 'price_entropy_1h', '5m_momentum', '15m_momentum',
            'spread_ratio', 'volatility_cluster', 'buy_sell_ratio', 'bid_ask_spread',
            'depth_imbalance', 'fractal_dimension', 'fib_retrace_38', 'fib_retrace_50',
            'order_flow_imbalance', 'rolling_kurtosis', 'lunar_phase'
        ]

    async def run(self):
        print("⏳ Loading and connecting to Binance WebSocket stream...")
        # Start the spinner task on its own
        spinner_handle = asyncio.create_task(spinner_task())

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
                            prediction = self.make_prediction(features)
                            self.handle_prediction(prediction, candle)
                            # Optionally, print a message to indicate processing is complete
                            print("\nWaiting for next candle...")
            except Exception as e:
                print(f"\nError: {str(e)}")
            finally:
                spinner_handle.cancel()  # Cancel the spinner task when done

    def make_prediction(self, features):
        """Convert features to model input and predict"""
        # Create DataFrame with correct column order
        X = pd.DataFrame([features], columns=self.feature_columns)
        
        # Handle missing data same as training
        X = X.ffill().fillna(0)
        
        # Make prediction
        proba = self.model.predict_proba(X)[0][1]
        return {
            'prediction': self.model.predict(X)[0],
            'probability': proba,
            'confidence': 'HIGH' if proba > 0.7 else 'MEDIUM' if proba > 0.5 else 'LOW'
        }

    def handle_prediction(self, prediction, candle):
        """Handle prediction results"""
        close_time = datetime.fromtimestamp(int(candle['T'])/1000)
        print(f"\n⏰ {close_time} | PEPE/USDT")
        print(f"🔮 Prediction: {'PUMP SIGNAL' if prediction['prediction'] else 'No pump'} "
              f"(Confidence: {prediction['confidence']} [{prediction['probability']:.2%}])")
        print(f"📈 Price: {candle['c']} | Volume: {candle['v']}")


if __name__ == "__main__":
    bot = LiveTradingBot()
    print("🚀 Starting PEPE/USDT Pump Detector...")
    asyncio.run(bot.run())