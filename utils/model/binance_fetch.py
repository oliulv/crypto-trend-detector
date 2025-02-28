import os
import requests
import pandas as pd
import time
import numpy as np
from datetime import datetime
import sys
from scipy.stats import kurtosis, entropy
from data_integrity import DataIntegrityChecker

# Basic parameters
symbol = "BTCUSDT"
interval = "1m"
start_date = "2017-07-15"
end_date = datetime.now().strftime("%Y-%m-%d")

# Label parameters
window_minutes = 60  # Prediction window in minutes
direction = "pump"   # "pump" or "dump"
threshold = 0.05     # Price change threshold (5%)

def format_window_string(minutes):
    """Convert minutes to human readable window string"""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h"

def create_label_type():
    """Generate label type string from parameters"""
    direction_str = direction.lower()
    threshold_str = str(int(threshold * 100)).zfill(2)
    return f"{direction_str}{threshold_str}"


def ensure_project_paths():
    """Ensure all required project directories exist"""
    # Get project root (two levels up from script)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at: {data_dir}")
    
    return project_root


# Terminal visuals for vibes
class TerminalVisuals:
    @staticmethod
    def fetch_animation(iteration, total):
        phases = ("⡿", "⣟", "⣯", "⣷", "⣾", "⣽", "⣻", "⢿")
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        sys.stdout.write(f"\r{phases[iteration % 8]} Fetching data | "
                        f"\033[90m{datetime.now().strftime('%H:%M:%S')}\033[0m | "
                        f"Progress: {percent}%")
        sys.stdout.flush()


def fetch_pepe_data_with_window_label(symbol, interval, start_date, end_date):
    base_url = "https://api.binance.com/api/v3/klines"
    data = []
    
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    print(f"Fetching {symbol} data")
    print(f"Period: {start_date} to {end_date}\n")
    
    total_pages = ((end_time - start_time) // (1000 * 60 * 1000)) + 1
    page_count = 0

    while start_time < end_time:
        # Visual update
        TerminalVisuals.fetch_animation(page_count, total_pages)
        page_count += 1
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"\n\033[31m✖ Critical failure: {response.status_code}\033[0m")
            break
            
        chunk = response.json()
        
        if not chunk:
            break
            
        data.extend(chunk)
        
        print(f"\nReceived {len(chunk)} records")
        print(f"\033[90mRange: {datetime.fromtimestamp(chunk[0][0]/1000)} → "
              f"{datetime.fromtimestamp(chunk[-1][0]/1000)}\033[0m")
        
        start_time = int(chunk[-1][0]) + 1
        time.sleep(0.1)
    
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    
    df = pd.DataFrame(data, columns=columns, dtype=float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate future window label
    df = create_label(df)
    
    return df


def create_label(df):
    """Create labels based on parameters"""
    # Calculate future price movement
    if direction == "pump":
        future_price = df['close'].shift(-1).rolling(window_minutes, min_periods=window_minutes).max()
        df['label'] = ((future_price - df['close']) / df['close'] >= threshold).astype(int)
    else:  # dump
        future_price = df['close'].shift(-1).rolling(window_minutes, min_periods=window_minutes).min()
        df['label'] = ((df['close'] - future_price) / df['close'] >= threshold).astype(int)
    
    # Remove rows without future data
    df = df.dropna(subset=['label'])
        
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'num_trades', 'taker_buy_base', 'taker_buy_quote', 'label']]


def add_extra_features(df):
    df = df.copy()
    
    df['1m_roc'] = df['close'].pct_change()
    df['30m_volatility'] = df['1m_roc'].rolling(30).std() * np.sqrt(30)
    df['volume_zscore_15m'] = (df['volume'] - df['volume'].rolling(15).mean()
                               ) / df['volume'].rolling(15).std()
    df['buy_pressure'] = df['taker_buy_base'] / df['volume'].replace(0, np.nan)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    # More price features
    df['1h_volatility'] = df['close'].pct_change().rolling(60).std() * np.sqrt(60)
    df['4h_ma_spread'] = df['close'] / df['close'].rolling(240).mean() - 1

    # More volume features
    df['volume_spike_15m'] = df['volume'] / df['volume'].rolling(15).mean()
    df['buy_sell_imbalance'] = df['taker_buy_base'] / df['volume']

    # More temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)

    # TECHNICAL INDICATORS:
    # RSI (past-only)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (past-only)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26

    # PRICE ACTION FEATURES:
    # Bollinger Bands %B
    window = 20
    df['ma_20'] = df['close'].rolling(window).mean()
    df['std_20'] = df['close'].rolling(window).std()
    df['bollinger_pct_b'] = (df['close'] - (df['ma_20'] - 2*df['std_20'])) / (4*df['std_20'])

    # Ichimoku Conversion Line (Tenkan-sen)
    period9_high = df['high'].rolling(9).max()
    period9_low = df['low'].rolling(9).min()
    df['ichimoku_conv'] = (period9_high + period9_low) / 2

    # VOLUME FEATURES:
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

    # Volume-Price Trend
    df['vpt'] = (df['volume'] * (df['close'].pct_change())).cumsum()

    # MARKET MICROSTRUCTURE:
    # Order Book Imbalance
    df['order_imbalance'] = (df['taker_buy_base'] - (df['volume'] - 
                             df['taker_buy_base'])) / df['volume']

    # Large Trade Dominance (using num_trades)
    df['avg_trade_size'] = df['volume'] / df['num_trades']
    df['large_trade_flag'] = (df['avg_trade_size'] > 
                              df['avg_trade_size'].rolling(60).quantile(0.9)).astype(int)   

    # ADVANCED VOLATILITY:
    # Average True Range (14-period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = np.maximum.reduce([high_low, high_close, low_close])
    df['atr'] = df['true_range'].rolling(14).mean()

    # MOMENTUM INDICATORS:
    # Stochastic Oscillator
    window = 14
    low_min = df['low'].rolling(window).min()
    high_max = df['high'].rolling(window).max()
    df['stochastic_%k'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = typical_price * df['volume']
    positive_mf = raw_mf.where(typical_price > typical_price.shift(), 0)
    negative_mf = raw_mf.where(typical_price < typical_price.shift(), 0)
    mf_ratio = positive_mf.rolling(14).sum() / negative_mf.rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + mf_ratio))

    #TEMPORAL PATTERNS:
    # Crypto Market Session Features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Asian/European/US Session Flags
    df['asian_session'] = df['timestamp'].dt.hour.between(0, 8).astype(int)
    df['us_session'] = df['timestamp'].dt.hour.between(13, 21).astype(int)

    # STATISTICAL FEATURES:
    # Rolling Skewness (30-min)
    df['return_skew'] = df['close'].pct_change().rolling(30, center=False).skew()

    # Entropy (60-min window)
    def price_entropy(x):
        hist = np.histogram(x, bins=10)[0]
        return entropy(hist/hist.sum())
    df['price_entropy_1h'] = df['close'].rolling(60).apply(price_entropy)

    # More Momentum Features:
    df['5m_momentum'] = df['close'].pct_change(5)  # 5-minute momentum
    df['15m_momentum'] = df['close'].pct_change(15)  # 15-minute momentum
    
    # More Liquidity Features:
    df['spread_ratio'] = (df['high'] - df['low']) / \
        df['volume'].replace(0, 1e-9)  # Spread-to-volume ratio

    # More Volatility Features:
    df['volatility_cluster'] = df['1m_roc'].rolling(30).var()  # 30-minute volatility

    # More Order Flow Features:
    df['buy_sell_ratio'] = df['taker_buy_base'] / \
        (df['volume'] - df['taker_buy_base']).replace(0, 1e-9)

    # Market Depth Featues:
    df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']
    df['depth_imbalance'] = df['taker_buy_base'] / (df['volume'] + 1e-9)

    # Chaotic Price Features:
    df['fractal_dimension'] = df['close'].rolling(30).apply(
        lambda x: (np.log(x.max() - x.min()) - np.log(x.std())) / np.log(30))

    # Advanced Technical Indicators
    df['fib_retrace_38'] = df['close'].rolling(20).apply(
        lambda x: x.max() - 0.382 * (x.max() - x.min()))
    df['fib_retrace_50'] = df['close'].rolling(20).apply(
        lambda x: x.max() - 0.5 * (x.max() - x.min()))

    # Market Microstructure
    df['order_flow_imbalance'] = (
        df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])) / df['volume']

    # Statistical Features
    df['rolling_kurtosis'] = df['close'].pct_change().rolling(30, center=False).apply(kurtosis)

    # Temporal Features
    df['lunar_phase'] = (df['timestamp'].dt.day % 29) / 29  # Simulated lunar phase

    return df


def generate_filename(symbol, interval, window, label_type, start_date, end_date):
    """Generate standardized filename for data storage
    
    Args:
        symbol (str): Trading pair symbol (e.g. 'PEPE')
        interval (str): Candle interval (e.g. '1m', '5m')
        window (str): Prediction window (e.g. '1h', '4h')
        label_type (str): Type of prediction (e.g. 'pump05')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    # Convert dates to compact format
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%y%m%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%y%m%d")
    
    return f"{symbol}_{interval}_{window}-{label_type}_{start}_{end}.csv"


if __name__ == "__main__":
    # Ensure project structure exists
    project_root = ensure_project_paths()
    
    # Fetch and process data
    df = fetch_pepe_data_with_window_label(symbol, interval, start_date, end_date)
    df = add_extra_features(df)

    # Generate window string and label type from parameters
    window = format_window_string(window_minutes)
    label_type = create_label_type()

    # Save to CSV with automatically generated parameters
    data_filename = generate_filename(
        symbol=symbol,
        interval=interval,
        window=window,
        label_type=label_type,
        start_date=start_date,
        end_date=end_date
    )
    
    data_path = os.path.join(project_root, 'data', data_filename)
    df.to_csv(data_path, index=False)

    # Final mystical output
    print("\nData processing complete")
    print(f"Total records: {len(df)}")
    print(f"Positive signals: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    print(f"Saved file: {data_filename} to: {data_path}")

    # Add divider and sleep
    print("\n" + "=" * 80)
    print("Running data integrity check...")
    time.sleep(5)
    print("=" * 80 + "\n")
    
    # Run data integrity check with full filepath
    checker = DataIntegrityChecker()
    checker.analyze_dataset(df, data_path)  # Use data_path instead of data_filename