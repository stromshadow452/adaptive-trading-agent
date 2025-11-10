import argparse, os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import TradingEnv

def make_env(path, window, fee_bps, slip_bps):
    def _init():
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        return TradingEnv(df, window=window, fee_bps=fee_bps, slip_bps=slip_bps)
    return _init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--tf', required=True)
    ap.add_argument('--timesteps', type=int, default=100000)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--fee_bps', type=float, default=1.0)
    ap.add_argument('--slip_bps', type=float, default=2.0)
    args = ap.parse_args()

    env = DummyVecEnv([make_env(args.processed, args.window, args.fee_bps, args.slip_bps)])
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='reports/tb')
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(f'models/checkpoints', exist_ok=True)
    out = f'models/checkpoints/{args.symbol}_{args.tf}.zip'
    model.save(out)
    print('✅ Saved model:', out)

if __name__ == '__main__':
    main()
