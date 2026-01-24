from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from src.rl.env import FinRLEnv
import os

class PPOAgent:
    """
    Stage 4: RL Brain -> PPO Agent
    Wraps Stable-Baselines3 PPO with Action Masking.
    """
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path)

    def train(self, df, total_timesteps=10000):
        env = FinRLEnv(df)
        env = ActionMasker(env, lambda e: e.valid_action_mask())
        
        self.model = PPO(MaskableActorCriticPolicy, env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        return self.model

    def predict(self, obs, action_masks):
        if not self.model:
            raise ValueError("Model not loaded or trained")
        
        action, _ = self.model.predict(obs, action_masks=action_masks)
        return action
