import torch.nn as nn
from torch import Tensor

import config


class FeedForward(nn.Module):
    """
    Linear layer followed by a non-linearity
    """
    n_embeddings: int
    multiplier: int

    def __init__(
            self, n_embeddings: int, 
            multiplier: int = config.DEFAULT_FEED_FORWARD_MULTIPLIER
        ) -> None:
        super().__init__()
        self.n_embeddings = n_embeddings
        self.multiplier = multiplier
        self.feed_forward = nn.Sequential(
            nn.Linear(self.n_embeddings, self.n_embeddings * multiplier),
            nn.ReLU(),
            nn.Linear(self.n_embeddings * multiplier, self.n_embeddings),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)
