"""Feature extractor for the two-part observation.

Lives in its own module because Stable-Baselines3 stores the extractor by its
import path inside the .zip: training and the agent node both have to be able
to import it from `social_rl.policy` or the checkpoint will not load.

The grid goes through three strided convolutions and the five-element vector is
concatenated afterwards, untouched. Feeding goal bearing and wheel speeds
through a convolution would only smear them across cells they have nothing to
do with, and they are already scaled to [-1, 1] by observation.py.
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SocialGridExtractor(BaseFeaturesExtractor):
    """(2, N, N) grid -> `cnn_features` numbers, concatenated with the vector."""

    def __init__(self, observation_space, cnn_features: int = 128):
        vector_dim = int(observation_space['vector'].shape[0])
        super().__init__(observation_space, features_dim=cnn_features + vector_dim)

        channels, height, width = observation_space['grid'].shape
        # Stride 2 three times: 32 -> 16 -> 8 -> 4. Deliberately small; the
        # whole point of the CNN here is translation equivariance, not depth,
        # and every extra layer is time taken from the 5 Hz control loop.
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, cnn_features), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        return torch.cat(
            [self.head(self.cnn(observations['grid'])), observations['vector']],
            dim=1)
