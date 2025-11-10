# tools/train_weekly_finrl.py
# Walk-forward retraining + evaluation + auto-promotion (FinRL + SB3)
# Usage (example):
#   python tools/train_weekly_finrl.py --symbol EURUSD --tf H1 --timesteps 300000 --algo ppo

import argparse, os, json, math, shutil, time, sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# --- FinRL / SB3 ---
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.env.env_stocktrading import StockTradingEnv

# Optional backtest stats (FinRL)
try:
    from finrl.trade.backtest import BackTestStats
    HAS_BACKTEST = True
except Exception:
    HAS_BACKTEST = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, required=True, help="e.g., EURUSD")
    p.add_argument("--tf", type=str, default="H1", help="e.g., H1, M15, H4")
    p.add_argument("--data_path", type=str, default=None,
                   help="CSV path; default: data/finrl/<SYMB>_<TF>_finrl.csv")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "ddpg", "sac", "td3"])
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--walk_train_days", type=int, default=90, help="lookback train window in days")
    p.add_argument("--walk_test_days", type=int, default=14, help="forward test window in days")
    p.add_argument("--stride_days", type=int, default=7, help="step size between walks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--promote_metric", type=str, default="sharpe", choices=["sharpe", "final_equity"])
    p.add_argument("--outdir", type=str, default=None,
                   help="base output dir; default: models/finrl/<SYMB>/<TF>")
    return p.parse_args()


def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    # Flexible date parsing
    date_col = "date" if "date" in df.columns else "time" if "time" in df.columns else None
    if date_col is None:
        raise ValueError("No date/time column found. Expected 'date' or 'time'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})
    # Minimal sanity
    needed = {"open", "high", "low", "close", "volume"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def make_env(df: pd.DataFrame):
    tech = [c for c in df.columns if c not in ["date", "open", "high", "low", "close", "volume"]]
    env_kwargs = dict(
        df=df,
        stock_dim=1,
        tech_indicator_list=tech,
        initial_amount=100_000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
    )
    return StockTradingEnv(**env_kwargs), tech


def train_agent(train_df: pd.DataFrame, algo: str, timesteps: int, seed: int):
    env_train, _ = make_env(train_df)
    agent = DRLAgent(env=env_train)
    model = agent.get_model(algo, seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f"finrl_{algo}", total_timesteps=timesteps)
    return trained


def run_episode(model, df: pd.DataFrame):
    # Evaluate on a fresh env built from df
    env_test, _ = make_env(df)
    obs = env_test.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
    # Extract account values
    try:
        account = env_test.save_asset_memory()
        # FinRL returns a df with 'date' and 'account_value'
        if isinstance(account, pd.DataFrame) and "account_value" in account.columns:
            acct_series = account["account_value"].astype(float).tolist()
        else:
            # fallback
            acct_series = env_test.account_memory
    except Exception:
        acct_series = getattr(env_test, "account_memory", [])
    return acct_series


def compute_metrics_from_equity(equity_curve):
    # Basic metrics if BackTestStats not available
    if len(equity_curve) < 3:
        return {"final_equity": float(equity_curve[-1]) if equity_curve else 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}
    eq = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    mean = np.mean(rets)
    std = np.std(rets) + 1e-12
    sharpe = (mean / std) * np.sqrt(252)  # daily-ish; rough scaling
    # Max drawdown
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    max_dd = float(drawdowns.min())
    return {"final_equity": float(eq[-1]), "sharpe": float(sharpe), "max_drawdown": max_dd}


def backtest_metrics(account_values):
    if HAS_BACKTEST:
        try:
            stats = BackTestStats(account_value=account_values)
            # BackTestStats prints & returns dict-like; try to map some keys
            # Safe extraction with defaults
            sharpe = float(stats.get("Sharpe Ratio", 0.0)) if isinstance(stats, dict) else 0.0
            calmar = float(stats.get("Calmar Ratio", 0.0)) if isinstance(stats, dict) else None
            mdd = stats.get("Max Drawdown", None)
            if isinstance(mdd, str) and "%" in mdd:
                try:
                    mdd = -abs(float(mdd.strip("%"))) / 100.0
                except:
                    mdd = None
            final_equity = float(account_values[-1]) if account_values else 0.0
            out = {"sharpe": sharpe, "calmar": calmar, "max_drawdown": mdd, "final_equity": final_equity}
            return out
        except Exception:
            pass
    # Fallback custom
    return compute_metrics_from_equity(account_values)


def rolling_windows(df: pd.DataFrame, train_days: int, test_days: int, stride_days: int):
    dates = df["date"]
    start = dates.min().normalize()
    end = dates.max().normalize()

    pointer = start + timedelta(days=train_days)
    while pointer + timedelta(days=test_days) <= end:
        train_start = pointer - timedelta(days=train_days)
        train_end = pointer
        test_end = pointer + timedelta(days=test_days)

        train_df = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
        test_df = df[(df["date"] >= train_end) & (df["date"] < test_end)].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            yield (train_start, train_end, test_end, train_df, test_df)

        pointer = pointer + timedelta(days=stride_days)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    tf = args.tf.upper()
    data_path = args.data_path or f"data/finrl/{symbol}_{tf}_finrl.csv"
    outdir = args.outdir or os.path.join("models", "finrl", symbol, tf)
    ensure_dirs(outdir)

    print(f"Symbol: {symbol}  TF: {tf}")
    print(f"Data: {data_path}")
    print(f"Algo: {args.algo}  Timesteps: {args.timesteps}")
    print(f"Walks: train {args.walk_train_days}d, test {args.walk_test_days}d, stride {args.stride_days}d")
    print(f"Outdir: {outdir}")

    df = load_data(data_path)

    # Walk-forward
    leaderboard = []
    best_ckpt_path = None
    best_metric_val = -1e18

    for i, (tr_start, tr_end, te_end, df_tr, df_te) in enumerate(
        rolling_windows(df, args.walk_train_days, args.walk_test_days, args.stride_days), start=1
    ):
        tag = f"walk{i:02d}_{tr_start.date()}_{tr_end.date()}__{tr_end.date()}_{te_end.date()}"
        run_dir = os.path.join(outdir, tag)
        ensure_dirs(run_dir)

        print(f"\n=== Walk {i} ===")
        print(f"Train: {tr_start.date()} -> {tr_end.date()}   |   Test: {tr_end.date()} -> {te_end.date()}")
        # Train
        model = train_agent(df_tr, args.algo, args.timesteps, args.seed)

        # Save checkpoint
        ckpt_path = os.path.join(run_dir, f"{args.algo}_policy_{args.timesteps}.zip")
        model.save(ckpt_path)
        print(f"Saved: {ckpt_path}")

        # Evaluate
        acct_values = run_episode(model, df_te)
        metrics = backtest_metrics(acct_values)
        metrics["walk_tag"] = tag
        metrics["algo"] = args.algo
        metrics["timesteps"] = args.timesteps
        metrics["n_test_steps"] = len(acct_values)
        save_json(os.path.join(run_dir, "metrics.json"), metrics)

        print("Metrics:", metrics)

        # Leaderboard metric
        metric_key = "sharpe" if args.promote_metric == "sharpe" else "final_equity"
        metric_val = float(metrics.get(metric_key, -1e18))
        leaderboard.append((metric_val, ckpt_path, metrics, run_dir))

        # Track best
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_ckpt_path = ckpt_path

    # Auto-promotion: copy best to stable/
    if best_ckpt_path is not None:
        stable_dir = os.path.join(outdir, "stable")
        ensure_dirs(stable_dir)
        promoted_path = os.path.join(stable_dir, f"{args.algo}_best.zip")
        shutil.copy2(best_ckpt_path, promoted_path)
        print(f"\n🏆 Promoted best model → {promoted_path}  ({args.promote_metric}={best_metric_val:.4f})")
        # Save leaderboard
        leaderboard_sorted = sorted(leaderboard, key=lambda x: x[0], reverse=True)
        report = {
            "promote_metric": args.promote_metric,
            "best_metric_value": best_metric_val,
            "best_checkpoint": best_ckpt_path,
            "runs": [
                {
                    "metric_value": mv,
                    "ckpt": ck,
                    "metrics": mx,
                    "run_dir": rd,
                }
                for (mv, ck, mx, rd) in leaderboard_sorted
            ],
        }
        save_json(os.path.join(outdir, "leaderboard.json"), report)
        print("Leaderboard saved.")
    else:
        print("No model was promoted (insufficient data windows?).")


if __name__ == "__main__":
    main()
