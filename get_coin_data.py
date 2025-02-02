import os
import requests
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import sys

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

class TerminalVisuals:
    @staticmethod
    def fetch_animation(iteration, total):
        phases = ("⡿", "⣟", "⣯", "⣷", "⣾", "⣽", "⣻", "⢿")
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        sys.stdout.write(f"\r\033[34m{phases[iteration % 8]}\033[0m Decrypting "
                         f"\033[33m{datetime.now().strftime('%H:%M:%S')}\033[0m | "
                         f"Epochs: \033[35m{total}\033[0m | "
                         f"Revelation: \033[36m{percent}%\033[0m")
        sys.stdout.flush()

    @staticmethod
    def crypto_header():
        print(r"""
    \033[35m
    ███████╗ ██████╗  █████╗ ██╗  ██╗
    ██╔════╝██╔═══██╗██╔══██╗██║ ██╔╝
    █████╗  ██║   ██║███████║█████╔╝ 
    ██╔══╝  ██║   ██║██╔══██║██╔═██╗ 
    ██║     ╚██████╔╝██║  ██║██║  ██╗
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    \033[0m
        """)

def fetch_pepe_data_with_window_label(symbol, interval, start_date, end_date):
    base_url = "https://api.binance.com/api/v3/klines"
    data = []
    
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    # Add mystical header
    TerminalVisuals.crypto_header()
    print(f"\033[35m•_ Starting PEPE data ritual ••_\033[0m")
    print(f"\033[33mTime Range: {start_date} → {end_date}\033[0m\n")
    
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
        
        # Add dramatic data reveal
        print(f"\n\033[32m✔ Received {len(chunk)} temporal fragments\033[0m")
        print(f"   First alignment: \033[36m{datetime.fromtimestamp(chunk[0][0]/1000)}\033[0m")
        print(f"   Last chronal signature:  \033[36m{datetime.fromtimestamp(chunk[-1][0]/1000)}\033[0m")
        
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
    df = create_window_label(df)
    
    return df

def create_window_label(df):
    # Calculate future max (ONLY FOR LABEL)
    future_max = df['close'].shift(-1).rolling(60, min_periods=60).max()
    
    # Calculate label directly
    df['label'] = ((future_max - df['close']) / df['close'] >= 0.05).astype(int)
    
    # Drop future-dependent columns
    df = df.drop(columns=['future_max', 'max_price_change_pct'], errors='ignore')
    
    # Remove rows without future data
    df = df.dropna(subset=['label'])
        
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'num_trades', 'taker_buy_base', 'taker_buy_quote', 'label']]

def add_window_features(df):
    # FIX: Prevent SettingWithCopyWarning
    df = df.copy()
    
    # FIX: Corrected pct_change() typo
    df['1m_roc'] = df['close'].pct_change()
    df['30m_volatility'] = df['1m_roc'].rolling(30).std() * np.sqrt(30)
    
    # FIX: Proper z-score calculation
    df['volume_zscore_15m'] = (df['volume'] - df['volume'].rolling(15).mean()) / df['volume'].rolling(15).std()
    
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
    df['order_imbalance'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])) / df['volume']

    # Large Trade Dominance (using num_trades)
    df['avg_trade_size'] = df['volume'] / df['num_trades']
    df['large_trade_flag'] = (df['avg_trade_size'] > df['avg_trade_size'].rolling(60).quantile(0.9)).astype(int)   

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
    df['return_skew'] = df['close'].pct_change().rolling(30).skew()

    # Entropy (60-min window)
    from scipy.stats import entropy
    def price_entropy(x):
        hist = np.histogram(x, bins=10)[0]
        return entropy(hist/hist.sum())
    df['price_entropy_1h'] = df['close'].rolling(60).apply(price_entropy)

    # More Momentum Features:
    df['5m_momentum'] = df['close'].pct_change(5)  # 5-minute momentum
    df['15m_momentum'] = df['close'].pct_change(15)  # 15-minute momentum
    
    # More Liquidity Features:
    df['spread_ratio'] = (df['high'] - df['low']) / df['volume'].replace(0, 1e-9)  # Spread-to-volume ratio

    # More Volatility Features:
    df['volatility_cluster'] = df['1m_roc'].rolling(30).var()  # 30-minute volatility

    # More Order Flow Features:
    df['buy_sell_ratio'] = df['taker_buy_base'] / (df['volume'] - df['taker_buy_base']).replace(0, 1e-9)
    return df

# Parameters
symbol = "PEPEUSDT"
interval = "1m"
start_date = "2023-05-20"
end_date = datetime.now().strftime("%Y-%m-%d")

# Fetch and process data
df = fetch_pepe_data_with_window_label(symbol, interval, start_date, end_date)
df = add_window_features(df)

# Save to CSV
save_path = os.path.join('data', f'PEPE_1hr_window_labels_{start_date}_to_{end_date}.csv')
df.to_csv(save_path, index=False)

# Final mystical output
print("\n\033[35m««« Data Conjuration Complete »»»\033[0m")
print(f"↳ Temporal Fragments: \033[36m{len(df)}\033[0m")
print(f"↳ Prophetic Visions: \033[32m{df['label'].sum()}\033[0m")
print(f"↳ Omen Ratio: \033[31m{df['label'].mean()*100:.2f}%\033[0m")
print(f"↳ Arcane Archive: \033[34m{save_path}\033[0m")
print("\033[33m\n«« The stars whisper their secrets »»\033[0m\n")