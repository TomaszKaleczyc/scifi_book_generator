import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    Linear layer followed by a non-linearity
    """
    n_embeddings: int

    def __init__(self, n_embeddings: int) -> None:
        super().__init__()
        self.n_embeddings = n_embeddings
        self.feed_forward = nn.Sequential(
            nn.Linear(self.n_embeddings, self.n_embeddings),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)
