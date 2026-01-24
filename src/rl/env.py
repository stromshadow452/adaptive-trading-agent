import gym
import numpy as np
import pandas as pd
from gym import spaces

class FinRLEnv(gym.Env):
    """
    Stage 4: RL Brain -> Environment
    Standard Gym Env with Action Masking support.
    """
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
        super(FinRLEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: [Close, RSI, SMA_Ratio, Balance, Position, Unrealized_PnL]
        # Simplified for demo. In prod, use feature_registry list + account state.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0 # 0=Flat, 1=Long, -1=Short (Simple)
        self.idx = 0
        self.entry_price = 0.0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        # Mock features for demo
        obs = np.array([
            row["close"], 
            row.get("rsi14", 50), 
            row.get("sma_ratio", 1.0),
            self.balance,
            self.position,
            (row["close"] - self.entry_price) * self.position if self.position != 0 else 0.0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        done = False
        row = self.df.iloc[self.idx]
        price = row["close"]
        reward = 0
        
        # Execute Action
        if action == 1: # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost
            elif self.position == -1: # Close Short
                pnl = (self.entry_price - price) / self.entry_price
                self.balance *= (1 + pnl - self.transaction_cost)
                self.position = 0
                reward = pnl

        elif action == 2: # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost
            elif self.position == 1: # Close Long
                pnl = (price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl - self.transaction_cost)
                self.position = 0
                reward = pnl
        
        self.idx += 1
        if self.idx >= len(self.df) - 1:
            done = True
            
        return self._get_obs(), reward, done, {}

    def valid_action_mask(self):
        # [Hold, Buy, Sell]
        mask = [True, True, True]
        
        # Cannot Buy if already Long or Short (Simple mode: 1 pos max)
        if self.position != 0: 
            mask[1] = False # Can't open new Long
            mask[2] = False # Can't open new Short (Wait, if Long, Sell closes it. If Short, Buy closes it.)
            
            # Refined Logic:
            if self.position == 1: # Long
                mask[1] = False # Can't Buy more
                mask[2] = True  # Can Sell (Close)
            elif self.position == -1: # Short
                mask[1] = True  # Can Buy (Close)
                mask[2] = False # Can't Sell more
                
        return np.array(mask)
