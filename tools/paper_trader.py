#!/usr/bin/env python
import argparse
import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import PPO, A2C

# Your env
from src.rl.finrl_forex_env import ForexEnv


def build_env(
    symbol: str,
    tf: str,
    data_root: str,
    start: str,
    end: str,
    cost_bps: float,
) -> ForexEnv:
    env = ForexEnv(
        data_root=data_root,
        symbol=symbol,
        tf_fast=tf,
        tf_base=tf,
        tf_slow="H1",
        start=start,
        end=end,
        cost_bps=cost_bps,
    )
    return env


def load_model(model_dir: str):
    # Prefer best_model if present
    best = os.path.join(model_dir, "best_model.zip")
    last = os.path.join(model_dir, "EURUSD_PPO_last")
    if os.path.exists(best):
        try:
            return PPO.load(best)
        except Exception:
            pass
    if os.path.exists(last):
        try:
            return PPO.load(last)
        except Exception:
            pass
    # Fallback: try generic SB3 load on directory
    return PPO.load(model_dir)


def _current_bar_ts(env: ForexEnv) -> Optional[pd.Timestamp]:
    """
    Get the *data* timestamp for the current bar from the env.
    Works whether the env exposes `current_step` or not.
    """
    if hasattr(env, "df") and isinstance(env.df.index, pd.DatetimeIndex) and len(env.df.index) > 0:
        # Try env.current_step if present
        step = getattr(env, "current_step", None)
        if step is None:
            # Some envs keep an internal pointer via something like env.i; fall back to last index
            return env.df.index[min(len(env.df.index) - 1, 0)]
        step = int(np.clip(step, 0, len(env.df.index) - 1))
        return env.df.index[step]
    return None


def run_paper(
    symbol: str,
    tf: str,
    data_root: str,
    model_dir: str,
    start: str,
    end: str,
    cost_bps: float,
    initial_equity: float,
    out_prefix: Optional[str],
    deterministic: bool,
    epsilon: Optional[float],
) -> Tuple[str, str, str]:
    env = build_env(symbol, tf, data_root, start, end, cost_bps)
    _ = env.reset()

    model = load_model(model_dir)

    # Outputs
    os.makedirs("reports", exist_ok=True)
    tag = out_prefix or f"paper_{symbol}_{tf}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    eq_path = os.path.join("reports", f"{tag}_equity.csv")
    tr_path = os.path.join("reports", f"{tag}_trades.csv")
    sm_path = os.path.join("reports", f"{tag}_summary.json")

    equity = initial_equity
    pos = 0  # -1 short, 0 flat, 1 long
    entry_price = None

    eq_rows = []
    tr_rows = []

    # Helper: record equity with DATA timestamp
    def log_equity(ts: Optional[pd.Timestamp]):
        ts = ts if ts is not None else pd.Timestamp.utcnow().tz_localize("UTC")
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC")
        eq_rows.append({"timestamp": ts.isoformat(), "equity": float(equity)})

    # Helper: record trade with DATA timestamp
    def log_trade(ts: Optional[pd.Timestamp], action: int, price: float, size: int):
        ts = ts if ts is not None else pd.Timestamp.utcnow().tz_localize("UTC")
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC")
        tr_rows.append(
            {
                "timestamp": ts.isoformat(),
                "action": action,          # 0 hold, 1 long, 2 short
                "price": float(price),
                "size": int(size),
                "position": int(pos),
                "equity": float(equity),
            }
        )

    # initial equity snapshot at first bar time
    log_equity(_current_bar_ts(env))

    terminated, truncated = False, False
    steps = 0

    # Small epsilon-greedy if requested
    def choose_action(obs):
        if epsilon is not None and np.random.rand() < epsilon:
            return int(np.random.randint(0, 3))
        a, _ = model.predict(obs, deterministic=deterministic)
        return int(a)

    obs = env._get_obs() if hasattr(env, "_get_obs") else env.reset()[0]

    while not (terminated or truncated):
        action = choose_action(obs)

        # Step the env
        res = env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            # SB3 v1 style
            obs, reward, done, info = res
            terminated, truncated = done, False

        # Get data timestamp for this bar
        bar_ts = _current_bar_ts(env)

        # Try to read current price from env (gracefully fall back)
        price = None
        if hasattr(env, "df") and len(env.df) > 0:
            # Use close price of current step if available, else last
            idx = getattr(env, "current_step", len(env.df) - 1)
            idx = int(np.clip(idx, 0, len(env.df) - 1))
            try:
                price = float(env.df["Close"].iloc[idx])
            except Exception:
                try:
                    price = float(env.df.iloc[idx]["Close"])
                except Exception:
                    price = None

        # Basic position accounting (toy, consistent with many paper_trader examples)
        # Action map: 0=hold, 1=go long, 2=go short
        if action == 1 and pos <= 0:
            # close short if any, then go long
            if pos < 0 and entry_price is not None and price is not None:
                # PnL from short
                equity += (entry_price - price) * 10000  # pip-scaled toy
            entry_price = price
            pos = 1
            log_trade(bar_ts, action, price if price is not None else np.nan, 1)

        elif action == 2 and pos >= 0:
            # close long if any, then go short
            if pos > 0 and entry_price is not None and price is not None:
                # PnL from long
                equity += (price - entry_price) * 10000
            entry_price = price
            pos = -1
            log_trade(bar_ts, action, price if price is not None else np.nan, -1)

        elif action == 0:
            # hold; no position change
            pass

        # mark-to-market equity update on each step
        if entry_price is not None and price is not None:
            if pos == 1:
                mtm = (price - entry_price) * 10000
            elif pos == -1:
                mtm = (entry_price - price) * 10000
            else:
                mtm = 0.0
            equity_mark = initial_equity + mtm
        else:
            equity_mark = equity  # unchanged

        # log equity at the bar time
        eq_rows.append({"timestamp": (bar_ts or pd.Timestamp.utcnow().tz_localize("UTC")).isoformat(),
                        "equity": float(equity_mark)})

        steps += 1

        # Stop at end of data if env signals via info
        if hasattr(info, "get") and info.get("end_of_data"):
            break

    # Final close: flat the position on last bar
    last_ts = _current_bar_ts(env)
    if pos != 0 and entry_price is not None and price is not None:
        if pos == 1:
            equity += (price - entry_price) * 10000
        elif pos == -1:
            equity += (entry_price - price) * 10000
        pos = 0
        log_trade(last_ts, 0, price if price is not None else np.nan, 0)

    # Write outputs
    eq_df = pd.DataFrame(eq_rows)
    tr_df = pd.DataFrame(tr_rows)

    # Ensure ISO strings (and sortable) + no tz loss
    for df in (eq_df, tr_df):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC")

    eq_df.sort_values("timestamp", inplace=True)
    tr_df.sort_values("timestamp", inplace=True)

    eq_df["timestamp"] = eq_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+00:00|0000)$", "+00:00", regex=True)
    tr_df["timestamp"] = tr_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+00:00|0000)$", "+00:00", regex=True)

    eq_df.to_csv(eq_path, index=False)
    tr_df.to_csv(tr_path, index=False)

    # Action histogram
    hist = {
        0: int((np.array([r["action"] for r in tr_rows]) == 0).sum()) if len(tr_rows) else 0,
        1: int((np.array([r["action"] for r in tr_rows]) == 1).sum()) if len(tr_rows) else 0,
        2: int((np.array([r["action"] for r in tr_rows]) == 2).sum()) if len(tr_rows) else 0,
    }

    # Exposure approximation
    total_steps = max(1, steps)
    flat_steps = hist.get(0, 0)
    long_steps = hist.get(1, 0)
    short_steps = hist.get(2, 0)
    flat_pct = 100.0 * flat_steps / max(1, (flat_steps + long_steps + short_steps))
    long_pct = 100.0 * long_steps / max(1, (flat_steps + long_steps + short_steps))
    short_pct = 100.0 * short_steps / max(1, (flat_steps + long_steps + short_steps))
    exposure_pct = 100.0 - flat_pct

    print(f"Saved equity to {eq_path}")
    print(f"Saved trades to {tr_path}")
    print(f"Action histogram: {hist}")
    print(f"Flat% = {flat_pct:.1f}% | Long% = {long_pct:.1f}% | Short% = {short_pct:.1f}% | Exposure% = {exposure_pct:.1f}%")

    summary = {
        "symbol": symbol,
        "tf": tf,
        "start": start,
        "end": end,
        "initial_equity": initial_equity,
        "final_equity": float(eq_df["equity"].iloc[-1]) if len(eq_df) else initial_equity,
        "action_histogram": hist,
        "exposure_pct": exposure_pct / 100.0,
        "equity_csv": eq_path,
        "trades_csv": tr_path,
    }
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary JSON: {sm_path}")
    return eq_path, tr_path, sm_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True, dest="tf")
    p.add_argument("--data_root", default="data/raw/forex")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--cost_bps", type=float, default=1.0)
    p.add_argument("--initial_equity", type=float, default=100000.0)
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--epsilon", type=float, default=None, help="epsilon-greedy prob for random action")
    return p.parse_args()


def main():
    args = parse_args()
    run_paper(
        symbol=args.symbol,
        tf=args.tf,
        data_root=args.data_root,
        model_dir=args.model_dir,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        initial_equity=args.initial_equity,
        out_prefix=args.out_prefix,
        deterministic=args.deterministic,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
