
import yfinance as yf
import pandas as pd


class YFinanceAdapter:
    def __init__(self):
        self.symbol_map = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "JPY=X",
            "XAGUSD": "SI=F",
        }

        self.interval_map = {
            "1H": "1h",
            "4H": "1h",
            "D1": "1h",
        }

    def get_latest_candles(
        self,
        symbol="EURUSD",
        timeframe="1H",
        bars=100
    ):
        yf_symbol = self.symbol_map[symbol]
        interval = self.interval_map[timeframe]

        df = yf.download(
            yf_symbol,
            period="14d",
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        # Keep required OHLCV columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df[required_cols].copy()

        # Normalize timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")

        return df.tail(bars)


if __name__ == "__main__":
    adapter = YFinanceAdapter()

    df = adapter.get_latest_candles(
        symbol="EURUSD",
        timeframe="1H",
        bars=10
    )

    print("\nLATEST EURUSD CANDLES:\n")
    print(df.tail())


