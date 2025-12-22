"""Custom neural network components for racecar SAC training."""

from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RaceCarMiniVGG(BaseFeaturesExtractor):
    """
    Compact VGG-style encoder with small channel counts.

    Uses two convs per block with 3x3 kernels, max pooling to downsample,
    and an adaptive avg pool to keep the flatten size small.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(
                "RaceCarMiniVGG expects (C, H, W) observations, "
                f"got shape {observation_space.shape}")

        channels, _, _ = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))
