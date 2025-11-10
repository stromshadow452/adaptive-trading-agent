#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train SB3 (PPO/A2C) on ForexEnv with multi-timeframe inputs.
Saves: best_model.zip + last checkpoint under --out.

Examples:
  python tools/train_finrl.py --symbol EURUSD \
    --data_root data/raw/forex_kaggle_multiTF \
    --tf_fast M15 --tf_base M15 --tf_slow H1 \
    --algo PPO --steps 400000 --eval_freq 10000 \
    --pos_bias_penalty 1e-5 --flat_penalty 0.0 \
    --start 2022-01-01 --end 2025-08-31 \
    --out models/finrl/ppo_EURUSD_M15_YYYYMMDD_HHMM
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# make local imports work
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rl.finrl_forex_env import ForexEnv


def build_env(
    data_root: str,
    symbol: str,
    tf_fast: str | None,
    tf_base: str | None,
    tf_slow: str | None,
    start: str | None,
    end: str | None,
    cost_bps: float,
    max_episode_steps: int | None,
    pos_bias_penalty: float,
    flat_penalty: float,
) -> ForexEnv:
    # Pick available timeframes if the requested ones aren’t present
    data_root = Path(data_root)
    symbol = symbol.upper()

    def tf_exists(tf: str) -> bool:
        return (data_root / f"{symbol}_{tf.upper()}.csv").exists()

    tfs_try = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

    def pick_any() -> str | None:
        for tf in tfs_try:
            if tf_exists(tf):
                return tf
        return None

    # Base
    tf_base = (tf_base or "M15").upper()
    if not tf_exists(tf_base):
        tf_base = pick_any()
        if tf_base is None:
            raise FileNotFoundError(f"No timeframe CSVs found for {symbol} under {data_root}")

    # Fast prefers same as base if missing
    tf_fast = (tf_fast.upper() if tf_fast else tf_base)
    if not tf_exists(tf_fast):
        tf_fast = tf_base if tf_exists(tf_base) else pick_any()

    # Slow prefers coarser than base, else base
    coarse_order = {"M1": "M5", "M5": "M15", "M15": "H1", "M30": "H1", "H1": "H4", "H4": "D1", "D1": "W1", "W1": "W1"}
    desired_slow = coarse_order.get(tf_base, "H1")
    tf_slow = (tf_slow.upper() if tf_slow else desired_slow)
    if not tf_exists(tf_slow):
        tf_slow = tf_base if tf_exists(tf_base) else pick_any()

    print(f"✅ Using data_root: {data_root}")
    print(f"✅ Timeframes -> fast:{tf_fast} base:{tf_base} slow:{tf_slow}")

    env = ForexEnv(
        data_root=str(data_root),
        symbol=symbol,
        tf_fast=tf_fast,
        tf_base=tf_base,
        tf_slow=tf_slow,
        start=start,
        end=end,
        cost_bps=cost_bps,
        pos_bias_penalty=pos_bias_penalty,
        flat_penalty=flat_penalty,
        max_episode_steps=max_episode_steps,
    )
    return env


def make_model(algo: str, env_vec, out_dir: Path):
    if algo.upper() == "PPO":
        model = PPO("MlpPolicy", env_vec, verbose=0)
    elif algo.upper() == "A2C":
        model = A2C("MlpPolicy", env_vec, verbose=0)
    else:
        raise ValueError("--algo must be PPO or A2C")

    out_dir.mkdir(parents=True, exist_ok=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--data_root", type=str, default="data/raw/forex_kaggle_multiTF")

    # timeframes
    ap.add_argument("--tf_fast", type=str, default=None)
    ap.add_argument("--tf_base", type=str, default="M15")
    ap.add_argument("--tf_slow", type=str, default=None)

    # date range
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)

    # algo & training
    ap.add_argument("--algo", type=str, choices=["PPO", "A2C"], default="PPO")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval_freq", type=int, default=10_000)
    ap.add_argument("--max_episode_steps", type=int, default=None)

    # costs & shaping
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--pos_bias_penalty", type=float, default=1e-5,
                    help="Per-bar penalty when in position to avoid 100% exposure.")
    ap.add_argument("--flat_penalty", type=float, default=0.0,
                    help="Per-bar penalty when flat (usually keep 0.0).")

    # output
    ap.add_argument("--out", type=str, required=True)

    args = ap.parse_args()
    out_dir = Path(args.out)

    # Build env and vectorize
    env = build_env(
        data_root=args.data_root,
        symbol=args.symbol,
        tf_fast=args.tf_fast,
        tf_base=args.tf_base,
        tf_slow=args.tf_slow,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        max_episode_steps=args.max_episode_steps,
        pos_bias_penalty=args.pos_bias_penalty,
        flat_penalty=args.flat_penalty,
    )
    env_vec = DummyVecEnv([lambda: env])

    # Model + eval callback
    model = make_model(args.algo, env_vec, out_dir)

       # Build evaluation environment
    eval_env = build_env(
        data_root=args.data_root,
        symbol=args.symbol,
        tf_fast=args.tf_fast,
        tf_base=args.tf_base,
        tf_slow=args.tf_slow,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        max_episode_steps=args.max_episode_steps,
        pos_bias_penalty=args.pos_bias_penalty,
        flat_penalty=args.flat_penalty,
    )
    eval_env_vec = DummyVecEnv([lambda: eval_env])

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env_vec,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    # ---- TRAIN ----
    print(f"\n🚀 Starting training for {args.steps:,} timesteps...")
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    print("\n✅ Training completed.\n")

    # ---- SAVE ----
    last_path = out_dir / f"{args.symbol}_{args.algo}_last"
    model.save(str(last_path))
    print(f"Saved: {last_path}")

    best_model_zip = out_dir / "best_model.zip"
    if best_model_zip.exists():
        print(f"Best model (if any): {best_model_zip}")
    else:
        print("⚠️ No best_model.zip found — possibly no eval improvement detected.")

if __name__ == "__main__":
    main()
