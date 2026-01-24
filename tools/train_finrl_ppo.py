"""
FinRL PPO Training Script for JARVIS Trading Agent

Trains a PPO model on EURUSD M15 data with exact 19-feature schema matching Primary ML Brain.

Usage:
    python tools/train_finrl_ppo.py --csv_path data/eurusd/EURUSD_M15.csv --model_out models/finrl/EURUSD_M15_policy.joblib
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.model_signing import sign_model_bytes, compute_feature_hash

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("[ERROR] stable-baselines3 or gymnasium not installed.")
    print("Install with: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# EXACT 19 features from Primary ML Brain (from executor.py TRAIN_FEATURES_DEFAULT)
REQUIRED_FEATURES = [
    "close", "ret", "sma5", "sma20", "sma_ratio", "sma50", "sma100", "sma_ratio_long",
    "atr14", "hl_range", "body", "ret_5", "ret_20", "rsi14", "boll_z", "atr_pct", "vol_norm", "hod", "dow"
]

# Expected feature hash (must match Primary model)
EXPECTED_FEATURE_HASH = "sha256:59207f8f82cf29243b0f580753de15c09632e3ca08092b2edda54e2fc3dc8c03"


class ForexTradingEnv(gym.Env):
    """
    Custom Forex Trading Environment for FinRL PPO Training
    
    Action Space: Discrete(3)
        -1: SELL
         0: HOLD
        +1: BUY
    
    Observation Space: Box(19,) - 19 features
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 19 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0
        self.total_profit = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.total_profit = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation (19 features)"""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        row = self.df.iloc[self.current_step]
        
        # Extract 19 features in exact order from REQUIRED_FEATURES
        obs = np.array([
            row["close"], row["ret"], row["sma5"], row["sma20"], row["sma_ratio"],
            row["sma50"], row["sma100"], row["sma_ratio_long"],
            row["atr14"], row["hl_range"], row["body"],
            row["ret_5"], row["ret_20"], row["rsi14"], row["boll_z"],
            row["atr_pct"], row["vol_norm"], row["hod"], row["dow"]
        ], dtype=np.float32)
        
        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def step(self, action):
        """
        Execute one step
        
        action: 0 (sell), 1 (hold), 2 (buy) from Discrete(3)
        Map to: -1 (sell), 0 (hold), +1 (buy)
        """
        # Map action from Discrete(3) to {-1, 0, +1}
        action_mapped = action - 1  # 0->-1, 1->0, 2->+1
        
        current_price = self.df.iloc[self.current_step]["close"]
        
        # Execute action
        reward = 0.0
        
        if action_mapped == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = current_price
        elif action_mapped == -1 and self.position == 0:  # SELL
            self.position = -1
            self.entry_price = current_price
        elif action_mapped == 0 and self.position != 0:  # CLOSE position
            if self.position == 1:  # Close long
                profit = (current_price - self.entry_price) / self.entry_price
            else:  # Close short
                profit = (self.entry_price - current_price) / self.entry_price
            
            reward = profit * 100  # Scale reward
            self.total_profit += profit
            self.position = 0
            self.entry_price = 0.0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, done, truncated, {}


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and compute 19 features using common_features.py
    
    Args:
        csv_path: Path to EURUSD M15 CSV
        
    Returns:
        DataFrame with all 19 features
    """
    print(f"[1/5] Loading data from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")  # Set timestamp as index for feature computation
    
    print(f"  Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    
    # Import feature computation from common_features.py
    print("[2/5] Computing 19 features using common_features.py...")
    try:
        from src.features.common_features import compute_features_from_ohlcv
        
        # Compute features (returns DataFrame with exact 19 features in correct order)
        df_features = compute_features_from_ohlcv(df)
        
        print(f"  After feature computation: {len(df_features)} candles")
        
        # Verify all 19 features present
        missing = set(REQUIRED_FEATURES) - set(df_features.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        return df_features
        
    except ImportError as e:
        print(f"  [ERROR] Could not import compute_features_from_ohlcv: {e}")
        print(f"  Make sure src/features/common_features.py exists")
        sys.exit(1)


def train_ppo_model(df: pd.DataFrame, total_timesteps: int = 100000) -> PPO:
    """
    Train PPO model on forex data
    
    Args:
        df: DataFrame with 19 features
        total_timesteps: Training timesteps
        
    Returns:
        Trained PPO model
    """
    print(f"[3/5] Training PPO model ({total_timesteps} timesteps)...")
    
    # Create environment
    env = ForexTradingEnv(df)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    print("  Training complete!")
    
    return model


def save_model_with_metadata(model: PPO, output_path: str, feature_hash: str):
    """
    Save PPO model with JARVIS metadata and HMAC signature
    
    Args:
        model: Trained PPO model
        output_path: Path to save .joblib file
        feature_hash: Feature hash for parity validation
    """
    print(f"[4/5] Saving model to {output_path}...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare metadata
    meta = {
        "symbol": "EURUSD",
        "tf": "M15",
        "feature_names": REQUIRED_FEATURES,
        "feature_hash": feature_hash,
        "feature_order_hash": feature_hash,  # Alias
        "model_type": "PPO",
        "model_name": "finrl_ppo",
        "trained_on": "EURUSD-M15",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
    }
    
    # Prepare payload (WITHOUT signatures first)
    payload = {
        "model": model.policy,  # Save policy only (lighter)
        "meta": meta
    }
    
    # Save FINAL version with metadata (but no signatures yet)
    joblib.dump(payload, output_path)
    
    # NOW read back the FINAL file and compute HMAC
    with open(output_path, "rb") as f:
        model_bytes = f.read()
    
    # Compute signatures on FINAL file
    sha256_hash, hmac_sig = sign_model_bytes(model_bytes)
    
    # Update metadata with signatures
    meta["sha256"] = sha256_hash
    meta["hmac"] = hmac_sig
    
    # Re-save with signatures (this changes the file, so HMAC won't match anymore!)
    # SOLUTION: Don't store signatures IN the file, only in .sig file
    # The HMAC should be computed on the file WITHOUT the HMAC field
    
    # Create .sig file with signatures
    sig_path = output_path + ".sig"
    with open(sig_path, "w") as f:
        json.dump({
            "sha256": sha256_hash,
            "hmac": hmac_sig
        }, f, indent=2)
    
    print(f"  Model saved: {output_path}")
    print(f"  Signature saved: {sig_path}")
    print(f"  Feature hash: {feature_hash}")
    print(f"  SHA256: {sha256_hash[:16]}...")
    print(f"  HMAC: {hmac_sig[:16]}...")


def main():
    parser = argparse.ArgumentParser(description="Train FinRL PPO model for JARVIS")
    parser.add_argument("--csv_path", required=True, help="Path to EURUSD M15 CSV")
    parser.add_argument("--model_out", default="models/finrl/EURUSD_M15_policy.joblib", help="Output model path")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FinRL PPO Training for JARVIS Protocol")
    print("=" * 80)
    
    # Compute and verify feature hash
    feature_hash = compute_feature_hash(REQUIRED_FEATURES)
    print(f"\n[FEATURE PARITY CHECK]")
    print(f"  Computed hash: {feature_hash}")
    print(f"  Expected hash: {EXPECTED_FEATURE_HASH}")
    
    if feature_hash != EXPECTED_FEATURE_HASH:
        print(f"\n[ERROR] Feature hash MISMATCH!")
        print(f"  This model will be REJECTED by JARVIS feature parity check.")
        print(f"  Verify that REQUIRED_FEATURES matches Primary model exactly.")
        sys.exit(1)
    
    print(f"  ✓ Feature hash matches Primary model!")
    
    # Load and prepare data
    df = load_and_prepare_data(args.csv_path)
    
    # Train model
    model = train_ppo_model(df, total_timesteps=args.timesteps)
    
    # Save with metadata
    save_model_with_metadata(model, args.model_out, feature_hash)
    
    print("\n[5/5] Training complete!")
    print(f"\nTo use this model with executor:")
    print(f"  python tools/executor.py \\")
    print(f"    --plans tests/plans \\")
    print(f"    --executions logs/rl_exec.csv \\")
    print(f"    --dry_run \\")
    print(f"    --csv_price_dir temp_prices \\")
    print(f"    --model_finrl_path {args.model_out} \\")
    print(f"    --verbose")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
