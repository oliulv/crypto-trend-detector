import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque


class LiveFeatureEngine:
    def __init__(self, lookback=60):
        self.buffer = deque(maxlen=lookback)
        self.initial_features = None
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                 'num_trades', 'taker_buy_base', 'taker_buy_quote']
        
        # Initialize stateful indicators
        self.ema12 = None
        self.ema26 = None
        self.prev_gain = 0
        self.prev_loss = 0

    def process_new_candle(self, candle):
        """Main processing pipeline for new candles"""
        # Convert to DataFrame for compatibility
        new_row = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(candle['t']/1000),
            'open': float(candle['o']),
            'high': float(candle['h']),
            'low': float(candle['l']),
            'close': float(candle['c']),
            'volume': float(candle['v']),
            'num_trades': float(candle['n']),
            'taker_buy_base': float(candle['V']),
            'taker_buy_quote': float(candle['Q'])
        }])

        # Update buffer
        self.buffer.append(new_row.iloc[0])
        df = pd.DataFrame(self.buffer)
        
        if len(df) < 30:  # Warm-up period
            return None

        # Calculate features incrementally
        return self.calculate_live_features(df)

    def calculate_live_features(self, df):
        """Replicate your feature engineering for live data"""
        features = {}
        
        # Basic features
        latest = df.iloc[-1]
        features['1m_roc'] = (latest['close'] - df.iloc[-2]['close'])/df.iloc[-2]['close']
        
        # Rolling volatility (last 30 minutes)
        returns = df['close'].pct_change().values
        features['30m_volatility'] = np.std(returns[-30:]) * np.sqrt(30)
        
        # Maintain state for RSI
        delta = latest['close'] - df.iloc[-2]['close']
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        self.prev_gain = (self.prev_gain * 13 + gain) / 14
        self.prev_loss = (self.prev_loss * 13 + loss) / 14
        features['rsi'] = 100 - (100 / (1 + (self.prev_gain / self.prev_loss))) if self.prev_loss != 0 else 50
        
        # MACD calculation
        self.ema12 = latest['close'] * 0.1538 + self.ema12 * 0.8462 if self.ema12 else latest['close']
        self.ema26 = latest['close'] * 0.0741 + self.ema26 * 0.9259 if self.ema26 else latest['close']
        features['macd'] = self.ema12 - self.ema26
        
        # Add other features following similar patterns...
        # (Implement remaining features using incremental calculations)
        
        return features