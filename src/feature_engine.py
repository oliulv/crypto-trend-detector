# feature_engine.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import deque
from scipy.stats import kurtosis, entropy


class LiveFeatureEngine:
    def __init__(self, lookback=1440, symbol="PEPEUSDT"):
        self.buffer = deque(maxlen=lookback)
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                 'num_trades', 'taker_buy_base', 'taker_buy_quote']

        # Preload historical data for 4h (240 candles)
        try:
            df_hist = LiveFeatureEngine.fetch_historical_binance(symbol, interval='1m', lookback=lookback)
            # Append each historical row (as a dict) into the buffer.
            for _, row in df_hist.iterrows():
                self.buffer.append(row.to_dict())
            print(f"✅ Loaded {len(self.buffer)} historical candles.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load historical data: {e}")

    @staticmethod
    def fetch_historical_binance(symbol, interval='1m', lookback=1440):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={lookback}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from Binance: {response.json()}")
        data = response.json()
        # Create DataFrame with required columns
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', 
            '_', 'num_trades', 'taker_buy_base', 'taker_buy_quote', '_'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                 'num_trades', 'taker_buy_base', 'taker_buy_quote']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Convert numeric columns to float
        for col in ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        return df

    def process_new_candle(self, candle):
        """Processes each new candle and returns the latest features once the buffer is full."""
        # Convert the incoming candle dict to a DataFrame row
        new_row = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(candle['t'] / 1000),
            'open': float(candle['o']),
            'high': float(candle['h']),
            'low': float(candle['l']),
            'close': float(candle['c']),
            'volume': float(candle['v']),
            'num_trades': float(candle['n']),
            'taker_buy_base': float(candle['V']),
            'taker_buy_quote': float(candle['Q'])
        }])
        # Add new data to the buffer
        self.buffer.append(new_row.iloc[0])
        df = pd.DataFrame(self.buffer)
        # Only return features once we have a full buffer.
        if len(df) < self.buffer.maxlen:
            return None
        return self.calculate_live_features(df)

    def calculate_live_features(self, df):
        """Calculate features from the full historical DataFrame without altering the original data."""
        # Work on a copy so that the original DataFrame remains intact
        temp = df.copy()
        features = {}

        # Directly pass through the latest basic data
        features['open'] = temp['open'].iloc[-1]
        features['high'] = temp['high'].iloc[-1]
        features['low'] = temp['low'].iloc[-1]
        features['close'] = temp['close'].iloc[-1]
        features['volume'] = temp['volume'].iloc[-1]
        features['num_trades'] = temp['num_trades'].iloc[-1]
        features['taker_buy_base'] = temp['taker_buy_base'].iloc[-1]
        features['taker_buy_quote'] = temp['taker_buy_quote'].iloc[-1]

        # 1-minute rate of change (ROC)
        temp['1m_roc'] = temp['close'].pct_change()
        features['1m_roc'] = temp['1m_roc'].iloc[-1]

        # 30-minute volatility: rolling std of 1m_roc over 30 periods
        temp['30m_volatility'] = temp['1m_roc'].rolling(30).std() * np.sqrt(30)
        features['30m_volatility'] = temp['30m_volatility'].iloc[-1]

        # Volume Z-Score over 15 minutes
        temp['volume_zscore_15m'] = (temp['volume'] - temp['volume'].rolling(15).mean()) / temp['volume'].rolling(15).std()
        features['volume_zscore_15m'] = temp['volume_zscore_15m'].iloc[-1]

        # Buy pressure: ratio from the most recent candle
        features['buy_pressure'] = temp['taker_buy_base'].iloc[-1] / (temp['volume'].iloc[-1] if temp['volume'].iloc[-1] != 0 else np.nan)

        # Temporal features (from last candle timestamp)
        last_timestamp = temp['timestamp'].iloc[-1]
        features['hour'] = last_timestamp.hour
        features['minute'] = last_timestamp.minute
        features['hour_sin'] = np.sin(2 * np.pi * last_timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * last_timestamp.hour / 24)

        # 1h volatility: 60-period rolling std of pct_change
        temp['1h_volatility'] = temp['close'].pct_change().rolling(60).std() * np.sqrt(60)
        features['1h_volatility'] = temp['1h_volatility'].iloc[-1]

        # 4h moving average spread: current close / 240-period rolling mean - 1
        temp['4h_ma'] = temp['close'].rolling(240).mean()
        features['4h_ma_spread'] = temp['close'].iloc[-1] / temp['4h_ma'].iloc[-1] - 1

        # Volume spike over 15 minutes
        temp['volume_spike_15m'] = temp['volume'] / temp['volume'].rolling(15).mean()
        features['volume_spike_15m'] = temp['volume_spike_15m'].iloc[-1]

        # Buy/sell imbalance: from the latest candle
        features['buy_sell_imbalance'] = temp['taker_buy_base'].iloc[-1] / temp['volume'].iloc[-1]

        # RSI calculation over 14 periods
        delta = temp['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        temp['rsi'] = 100 - (100 / (1 + rs))
        features['rsi'] = temp['rsi'].iloc[-1]

        # MACD: difference between 12 and 26 period EMAs
        temp['ema12'] = temp['close'].ewm(span=12, adjust=False).mean()
        temp['ema26'] = temp['close'].ewm(span=26, adjust=False).mean()
        temp['macd'] = temp['ema12'] - temp['ema26']
        features['macd'] = temp['macd'].iloc[-1]

        # Bollinger Bands %B (using window=20)
        temp['ma_20'] = temp['close'].rolling(20).mean()
        temp['std_20'] = temp['close'].rolling(20).std()
        temp['bollinger_pct_b'] = (temp['close'] - (temp['ma_20'] - 2 * temp['std_20'])) / (4 * temp['std_20'])
        features['bollinger_pct_b'] = temp['bollinger_pct_b'].iloc[-1]

        # Ichimoku Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
        temp['ichimoku_conv'] = (temp['high'].rolling(9).max() + temp['low'].rolling(9).min()) / 2
        features['ichimoku_conv'] = temp['ichimoku_conv'].iloc[-1]

        # On-Balance Volume (OBV)
        temp['obv'] = (np.sign(temp['close'].diff()) * temp['volume']).fillna(0).cumsum()
        features['obv'] = temp['obv'].iloc[-1]

        # Volume-Price Trend (VPT)
        temp['vpt'] = (temp['volume'] * temp['close'].pct_change()).fillna(0).cumsum()
        features['vpt'] = temp['vpt'].iloc[-1]

        # Order imbalance: (taker_buy_base - (volume - taker_buy_base)) / volume
        features['order_imbalance'] = (temp['taker_buy_base'].iloc[-1] - (temp['volume'].iloc[-1] - temp['taker_buy_base'].iloc[-1])) / temp['volume'].iloc[-1]

        # Average trade size and large trade flag (using rolling quantile over 60 periods)
        temp['avg_trade_size'] = temp['volume'] / temp['num_trades'].replace(0, np.nan)
        features['avg_trade_size'] = temp['avg_trade_size'].iloc[-1]
        quantile = temp['avg_trade_size'].rolling(60).quantile(0.9)
        features['large_trade_flag'] = 1 if temp['avg_trade_size'].iloc[-1] > quantile.iloc[-1] else 0

        # True range and ATR (14-period)
        high_low = temp['high'] - temp['low']
        high_close = (temp['high'] - temp['close'].shift()).abs()
        low_close = (temp['low'] - temp['close'].shift()).abs()
        temp['true_range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        temp['atr'] = temp['true_range'].rolling(14).mean()
        features['true_range'] = temp['true_range'].iloc[-1]
        features['atr'] = temp['atr'].iloc[-1]

        # Stochastic Oscillator %K (window=14)
        low_min = temp['low'].rolling(14).min()
        high_max = temp['high'].rolling(14).max()
        temp['stochastic_%k'] = 100 * (temp['close'] - low_min) / (high_max - low_min)
        features['stochastic_%k'] = temp['stochastic_%k'].iloc[-1]

        # Money Flow Index (MFI) calculation over 14 periods
        typical_price = (temp['high'] + temp['low'] + temp['close']) / 3
        raw_mf = typical_price * temp['volume']
        positive_mf = raw_mf.where(typical_price > typical_price.shift(), 0)
        negative_mf = raw_mf.where(typical_price < typical_price.shift(), 0)
        mf_ratio = positive_mf.rolling(14).sum() / negative_mf.rolling(14).sum()
        temp['mfi'] = 100 - (100 / (1 + mf_ratio))
        features['mfi'] = temp['mfi'].iloc[-1]

        # Day of week and session flags
        features['day_of_week'] = temp['timestamp'].iloc[-1].dayofweek
        features['is_weekend'] = 1 if temp['timestamp'].iloc[-1].dayofweek in [5, 6] else 0
        features['asian_session'] = 1 if 0 <= temp['timestamp'].iloc[-1].hour < 8 else 0
        features['us_session'] = 1 if 13 <= temp['timestamp'].iloc[-1].hour < 21 else 0

        # Return skew: rolling skewness of close pct_change over 30 periods
        temp['return_skew'] = temp['close'].pct_change().rolling(30).skew()
        features['return_skew'] = temp['return_skew'].iloc[-1]

        # Price entropy: using a 60-period window
        def price_entropy(x):
            hist = np.histogram(x, bins=10)[0]
            return entropy(hist / hist.sum())
        temp['price_entropy_1h'] = temp['close'].rolling(60).apply(price_entropy, raw=False)
        features['price_entropy_1h'] = temp['price_entropy_1h'].iloc[-1]

        # Momentum features: 5-minute and 15-minute momentum
        temp['5m_momentum'] = temp['close'].pct_change(5)
        features['5m_momentum'] = temp['5m_momentum'].iloc[-1]
        temp['15m_momentum'] = temp['close'].pct_change(15)
        features['15m_momentum'] = temp['15m_momentum'].iloc[-1]

        # Spread ratio: (high - low) / volume
        temp['spread_ratio'] = (temp['high'] - temp['low']) / temp['volume'].replace(0, 1e-9)
        features['spread_ratio'] = temp['spread_ratio'].iloc[-1]

        # Volatility cluster: variance of 1m_roc over 30 periods
        temp['volatility_cluster'] = temp['1m_roc'].rolling(30).var()
        features['volatility_cluster'] = temp['volatility_cluster'].iloc[-1]

        # Buy-sell ratio: taker_buy_base / (volume - taker_buy_base)
        features['buy_sell_ratio'] = temp['taker_buy_base'].iloc[-1] / ((temp['volume'].iloc[-1] - temp['taker_buy_base'].iloc[-1]) or 1e-9)

        # Bid-ask spread: (high - low) / close
        features['bid_ask_spread'] = (temp['high'].iloc[-1] - temp['low'].iloc[-1]) / temp['close'].iloc[-1]

        # Depth imbalance: taker_buy_base / volume
        features['depth_imbalance'] = temp['taker_buy_base'].iloc[-1] / (temp['volume'].iloc[-1] + 1e-9)

        # Fractal dimension: rolling apply on 30 periods
        temp['fractal_dimension'] = temp['close'].rolling(30).apply(
            lambda x: (np.log(x.max() - x.min()) - np.log(x.std())) / np.log(30))
        features['fractal_dimension'] = temp['fractal_dimension'].iloc[-1]

        # Fibonacci retracement levels (38% and 50%) over a 20-period window
        temp['fib_retrace_38'] = temp['close'].rolling(20).apply(lambda x: x.max() - 0.382 * (x.max() - x.min()))
        features['fib_retrace_38'] = temp['fib_retrace_38'].iloc[-1]
        temp['fib_retrace_50'] = temp['close'].rolling(20).apply(lambda x: x.max() - 0.5 * (x.max() - x.min()))
        features['fib_retrace_50'] = temp['fib_retrace_50'].iloc[-1]

        # Order flow imbalance (again)
        features['order_flow_imbalance'] = (temp['taker_buy_base'].iloc[-1] - (temp['volume'].iloc[-1] - temp['taker_buy_base'].iloc[-1])) / temp['volume'].iloc[-1]

        # Rolling kurtosis of close pct_change over 30 periods
        temp['rolling_kurtosis'] = temp['close'].pct_change().rolling(30).apply(kurtosis)
        features['rolling_kurtosis'] = temp['rolling_kurtosis'].iloc[-1]

        # Lunar phase (simulated): (day % 29) / 29
        features['lunar_phase'] = (temp['timestamp'].iloc[-1].day % 29) / 29

        return features
