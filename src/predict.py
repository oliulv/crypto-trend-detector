import websockets
import asyncio
import json
import pandas as pd
from datetime import datetime
from feature_engine import LiveFeatureEngine


# Load your trained model here
# from your_model_file import load_model
# model = load_model('path/to/model')


async def live_predictions():
    feature_engine = LiveFeatureEngine()
    
    uri = "wss://stream.binance.com:9443/ws/pepeusdt@kline_1m"
    async with websockets.connect(uri) as ws:
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                candle = data['k']
                
                if candle['x']:  # Only process closed candles
                    features = feature_engine.process_new_candle(candle)
                    
                    if features is not None:
                        # Convert to model input format
                        model_input = pd.DataFrame([features]).values.reshape(1, -1)
                        
                        # Get prediction
                        # prediction = model.predict(model_input)
                        # print(f"Model Prediction: {prediction}")
                        
                        print(f"Processed candle at {datetime.fromtimestamp(candle['T']/1000)}")
                        print(features)

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

if __name__ == "__main__":
    # Run the live bot
    asyncio.get_event_loop().run_until_complete(live_trading_bot())