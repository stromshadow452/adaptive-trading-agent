# tools/eval_finrl.py
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.trade.backtest import BackTestStats
import pandas as pd
from finrl.trade.backtest import BackTestStats
from finrl.env.env_stocktrading import StockTradingEnv
import pandas as pd

df = pd.read_csv("data/finrl/EURUSD_H1_finrl.csv", parse_dates=["date"])
tech = [c for c in df.columns if c not in ["date","open","high","low","close","volume"]]
env = StockTradingEnv(df=df, stock_dim=1, tech_indicator_list=tech)

print("Running backtest...")
BackTestStats(account_value=env.account_memory)  # prints Sharpe, MaxDD, etc.
