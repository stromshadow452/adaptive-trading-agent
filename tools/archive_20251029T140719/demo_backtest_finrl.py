# tools/demo_backtest_finrl.py
import pandas as pd
import matplotlib.pyplot as plt
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO

DATA_PATH = "data/finrl/EURUSD_H1_finrl.csv"
MODEL_PATH = "models/finrl/EURUSD/policy_100k.zip"  # change if your filename differs

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]
    df["sma10"] = c.rolling(10, min_periods=1).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = up / (down.replace(0, 1e-12))
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df.bfill().ffill()

# 1) Data + indicators
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["tic"] = "EURUSD"
df = add_indicators(df)
tech = ["sma10", "ema20", "rsi14"]

# 2) Env (per-asset lists)
env_kwargs = dict(
    df=df,
    stock_dim=1,
    hmax=[100],
    initial_amount=100_000,
    buy_cost_pct=[0.001],
    sell_cost_pct=[0.001],
    num_stock_shares=[0],
    tech_indicator_list=tech,
    reward_scaling=1e-4,
    state_space=1 + 2 + len(tech),
    action_space=1,
)
raw_env = StockTradingEnv(**env_kwargs)

# 3) Load model, attach env (SB3 will wrap with DummyVecEnv)
# ignore FloatSchedule warnings from older/newer SB3 versions
model = PPO.load(MODEL_PATH, device="cpu")
model.set_env(raw_env)
vec_env = model.get_env()
base_env = vec_env.envs[0]  # underlying FinRL env to read internals

# 4) Rollout (handle tuple obs from Gymnasium)
obs = vec_env.reset()
if isinstance(obs, tuple):  # Gymnasium-style reset returns (obs, info)
    obs = obs[0]

account_values = []
for _ in range(len(df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    step_out = vec_env.step(action)

    # step() may return 4-tuple (obs, reward, done, info) OR 5-tuple with truncation
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs, rewards, dones, infos = step_out
    else:
        # Fallback: unpack first 4 items
        obs, rewards, dones, infos = step_out[:4]

    if isinstance(obs, tuple):  # just in case
        obs = obs[0]

    # Read portfolio value from the base env
    pv = getattr(base_env, "portfolio_value", None)
    if pv is None and getattr(base_env, "asset_memory", None):
        pv = base_env.asset_memory[-1]
    if pv is None:
        # last-resort reconstruction
        cash = getattr(base_env, "cash", None)
        holdings = getattr(base_env, "stock_holdings", [0])
        prices = getattr(base_env, "stock_prices", [0.0])
        if cash is not None:
            pv = cash + holdings[0] * prices[0]
    account_values.append(float(pv) if pv is not None else 0.0)

    # Handle termination for VecEnv
    done_flag = dones[0] if isinstance(dones, (list, tuple)) else dones
    if done_flag:
        break

# 5) Plot equity curve
if account_values:
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"][: len(account_values)], account_values, label="Account Value ($)")
    plt.title("FinRL EURUSD Agent Backtest Performance")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    start_val = 100000.0
    end_val = account_values[-1]
    print(f"✅ Backtest finished. Start: {start_val:.2f}  Final: {end_val:.2f}  Δ: {end_val - start_val:.2f}")
else:
    print("Backtest produced no steps (empty curve).")
