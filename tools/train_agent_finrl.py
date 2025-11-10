# tools/train_agent_finrl.py
import os
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

DATA_PATH = "data/finrl/EURUSD_H1_finrl.csv"
MODEL_DIR = "models/finrl/EURUSD"
TIMESTEPS = 100_000  # start small; increase later

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]

    # Simple technicals to ensure state size is sufficient
    df["sma10"] = c.rolling(10, min_periods=1).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()

    # RSI14
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = up / (down.replace(0, 1e-12))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # Fill NaNs created by indicators
    df = df.bfill().ffill()
    return df

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # FinRL stock env expects a 'tic' column even with one instrument
    if "tic" not in df.columns:
        df["tic"] = "EURUSD"

    # Add minimal indicators
    df = add_indicators(df)
    tech = ["sma10", "ema20", "rsi14"]

    stock_dim = 1
    action_space = stock_dim
    state_space = 1 + 2 * stock_dim + len(tech) * stock_dim  # cash + price/holdings + tech
    num_stock_shares = [0] * stock_dim
    hmax_list = [100] * stock_dim              # per-asset max units per order
    buy_cost_list = [0.001] * stock_dim        # per-asset fees
    sell_cost_list = [0.001] * stock_dim

    env_train = StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax_list,                         # ✅ list (this build indexes per asset)
        initial_amount=100_000,
        buy_cost_pct=buy_cost_list,             # ✅ list
        sell_cost_pct=sell_cost_list,           # ✅ list
        num_stock_shares=num_stock_shares,
        tech_indicator_list=tech,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
    )

    policy_kwargs = dict(net_arch=[64, 64])
    model_kwargs = dict(
        device="cpu",       # SB3 accepts 'cpu'/'cuda' etc. (no DirectML string)
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.0,
        n_epochs=5,
    )
    print(model_kwargs)

    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
    trained = agent.train_model(model=model, tb_log_name="finrl_ppo", total_timesteps=TIMESTEPS)

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, f"policy_{TIMESTEPS//1000}k.zip")
    trained.save(out_path)
    print(f"✅ FinRL PPO model saved to: {out_path}")

if __name__ == "__main__":
    main()
