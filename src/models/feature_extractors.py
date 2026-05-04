from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BayesGatedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 128,
        activation: str = "tanh",
        gating: bool = False,
        posterior_var_index: int = -4,
        gating_mode: str = "sigmoid",
        gate_init_std: float = 0.0,
        attention: bool = False,
        aux_feature_indices: list[int] | None = None,
    ):
        super().__init__(observation_space, features_dim)
        act_cls = nn.ReLU if activation.lower() == "relu" else nn.Tanh
        self.backbone = nn.Sequential(
            nn.LayerNorm(observation_space.shape[0]),
            nn.Linear(observation_space.shape[0], hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, features_dim),
            act_cls(),
        )
        self.gating = gating
        self.gating_mode = gating_mode
        self.posterior_var_index = posterior_var_index
        self.attention = attention
        self.aux_feature_indices = list(aux_feature_indices or [])
        self.gate_weight = nn.Parameter(torch.zeros(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        aux_dim = len(self.aux_feature_indices)
        self.attention_logits = nn.Linear(features_dim + aux_dim, aux_dim) if self.attention and aux_dim > 0 else None
        self.attention_projection = nn.Linear(aux_dim, features_dim) if self.attention and aux_dim > 0 else None
        if gate_init_std > 0.0:
            nn.init.normal_(self.gate_weight, mean=0.0, std=gate_init_std)
            nn.init.normal_(self.gate_bias, mean=0.0, std=gate_init_std)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(observations)
        if self.attention and self.attention_logits is not None and self.attention_projection is not None:
            aux = observations[:, self.aux_feature_indices]
            logits = self.attention_logits(torch.cat([hidden, aux], dim=1))
            weights = torch.softmax(logits, dim=1)
            hidden = hidden + self.attention_projection(weights * aux)
        if not self.gating:
            return hidden
        posterior_var = observations[:, self.posterior_var_index].unsqueeze(1)
        if self.gating_mode == "linear":
            gate = 1.0 + self.gate_weight * posterior_var + self.gate_bias
        else:
            gate = torch.sigmoid(self.gate_weight * posterior_var + self.gate_bias)
        return hidden * gate
