import os
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd


class FinnhubAdapter:
    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            raise ValueError("FINNHUB_API_KEY not found")

        self.client = finnhub.Client(api_key=api_key)

        self.symbol_map = {
            "EURUSD": "OANDA:EUR_USD",
            "GBPUSD": "OANDA:GBP_USD",
            "USDJPY": "OANDA:USD_JPY",
            "XAGUSD": "OANDA:XAG_USD",
        }

        self.resolution_map = {
            "1H": "60",
            "4H": "240",
            "D1": "D",
        }

    def get_latest_candles(self, symbol="EURUSD", timeframe="1H", bars=50):
        finnhub_symbol = self.symbol_map[symbol]
        resolution = self.resolution_map[timeframe]

        now = int(time.time())
        start = now - (bars * 3600 * 4)

        data = self.client.forex_candles(
            finnhub_symbol,
            resolution,
            start,
            now
        )

        if data.get("s") != "ok":
            raise ValueError(f"Finnhub API error: {data}")

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"],
        })

        df.set_index("timestamp", inplace=True)

        return df


if __name__ == "__main__":
    adapter = FinnhubAdapter()

    df = adapter.get_latest_candles("EURUSD", "1H", bars=10)

    print(df.tail())
