import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    """
    Module representing a single self 
    attention head operations
    """
    n_embedding: int
    head_size: int
    block_size: int

    def __init__(
            self,
            n_embedding: int,
            head_size: int,
            block_size: int
        ) -> None:
        super(SelfAttentionHead, self).__init__()
        self.n_embedding = n_embedding
        self.head_size = head_size
        self.block_size = block_size

        self.key_table = nn.Linear(self.n_embedding, self.head_size, bias=False)
        self.query_table = nn.Linear(self.n_embedding, self.head_size, bias=False)
        self.value_table = nn.Linear(self.n_embedding, self.head_size, bias=False)
        self._set_tril()

    def _set_tril(self) -> None:
        """
        Sets the lower triangular matrix for masking
        """
        ones = torch.ones(self.block_size, self.block_size)
        tril = torch.tril(ones)
        self.register_buffer('tril', tril)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        key = self.key_table(x)  # (B,T,C)
        query = self.query_table(x)  # (B,T,C)
        value = self.value_table(x)  # (B,T,C)

        weights = query @ key.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        weights = F.softmax(weights, dim=-1)
        output = weights @ value  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return output
