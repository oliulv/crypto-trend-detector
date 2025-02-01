import requests
import pandas as pd
import time

def fetch_historical_prices(coin_id, days=30):
    """
    Fetch daily/historical data for the given coin using CoinGecko's API.
    `days` can be an integer or 'max' for full data.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days
    }
    response = requests.get(url, params=params)
    data = response.json()

    # data['prices'] is a list of [timestamp, price]
    prices = data['prices']  
    # You can also explore 'market_caps', 'total_volumes' if you want

    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    # Convert timestamp (in ms) to a readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

if __name__ == "__main__":
    df_prices = fetch_historical_prices(coin_id="pepe", days="max")
    df_prices.to_csv("doge_prices.csv", index=False)
    print("Data saved to doge_prices.csv")
