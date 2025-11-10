"""
LSTM Feature Extractor compatible with Stable-Baselines3 (FinRL).
- Use in policy kwargs:
  policy_kwargs=dict(
      features_extractor_class=LSTMFeatureExtractor,
      features_extractor_kwargs=dict(input_dim=feat_dim, hidden_size=64, num_layers=1, proj_dim=64)
  )
"""
from typing import Tuple
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_dim: int = None,
                 hidden_size: int = 64, num_layers: int = 1, proj_dim: int = 64, dropout: float = 0.0):
        # observation_space may be Box(shape=(T, F)) or (F,) flattened sequence provided via env wrapper
        if hasattr(observation_space, "shape"):
            obs_shape = observation_space.shape
            if len(obs_shape) == 1:
                # flattened, must also pass input_dim via kwargs; treat as single timestep
                T, F = 1, input_dim or obs_shape[0]
            elif len(obs_shape) == 2:
                T, F = obs_shape[0], obs_shape[1]
            else:
                raise ValueError(f"Unsupported obs shape: {obs_shape}")
        else:
            T, F = 1, int(input_dim)

        self.seq_len = T
        self.feat_dim = F
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features = proj_dim

        super().__init__(observation_space, features_dim=proj_dim)

        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations shapes:
        - [B, T, F]  (preferred)
        - [B, F]     (will be unsqueezed to T=1)
        """
        x = observations
        if x.dim() == 2:
            x = x.unsqueeze(1)  # B,1,F
        if x.size(-1) != self.feat_dim:
            raise RuntimeError(f"Expected last dim {self.feat_dim}, got {x.size(-1)}")
        out, (h_n, c_n) = self.lstm(x)      # h_n: [num_layers, B, H]
        last = h_n[-1]                       # [B, H]
        z = self.proj(last)                  # [B, proj_dim]
        return z
