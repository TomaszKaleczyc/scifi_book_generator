import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    """
    Module representing a single self 
    attention head operations
    """
    n_embeddings: int
    head_size: int
    block_size: int
    dropout_probability: float

    def __init__(
            self,
            n_embeddings: int,
            head_size: int,
            block_size: int,
            dropout_probability: float
        ) -> None:
        super(SelfAttentionHead, self).__init__()
        self.n_embeddings = n_embeddings
        self.head_size = head_size
        self.block_size = block_size
        self.dropout_probability = dropout_probability

        self.key_table = nn.Linear(self.n_embeddings, self.head_size, bias=False)
        self.query_table = nn.Linear(self.n_embeddings, self.head_size, bias=False)
        self.value_table = nn.Linear(self.n_embeddings, self.head_size, bias=False)
        self._set_tril()
        self.dropout = nn.Dropout(p=self.dropout_probability)

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
        weights = self.dropout(weights)

        output = weights @ value  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return output
