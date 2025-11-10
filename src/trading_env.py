import gymnasium as gym
from gymnasium import spaces
import numpy as np

def make_feature_matrix(df):
    cols = ['ret','ema8','ema21','ema50','rsi14','atr14','vol_ratio']
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0.0
    return df2[cols].fillna(0.0)

class TradingEnv(gym.Env):
    def __init__(self, df, window=64, fee_bps=1.0, slip_bps=2.0):
        super().__init__()
        self.window = window
        self.fee = (fee_bps + slip_bps) / 1e4
        self.df = make_feature_matrix(df)
        self.prices = df['Close'].values
        self.ptr = window
        self.pos = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window, self.df.shape[1]), dtype=np.float32)

    def _obs(self):
        block = self.df.iloc[self.ptr-self.window:self.ptr].values.astype('float32')
        return block

    def reset(self, seed=None, options=None):
        self.ptr = self.window
        self.pos = 0
        return self._obs(), {}

    def step(self, action):
        mapping = {0:-1, 1:0, 2:1}
        new_pos = mapping.get(int(action), 0)
        prev = self.pos
        if self.ptr >= len(self.df)-1:
            return self._obs(), 0.0, True, False, {}
        r = np.log(self.prices[self.ptr+1]) - np.log(self.prices[self.ptr])
        trade_cost = self.fee if new_pos != prev else 0.0
        reward = new_pos * r - trade_cost
        self.pos = new_pos
        self.ptr += 1
        done = (self.ptr >= len(self.df)-1)
        return self._obs(), float(reward), done, False, {'pos': self.pos}
