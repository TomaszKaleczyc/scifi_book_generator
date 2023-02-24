import torch
import torch.nn as nn
from torch import Tensor

from .self_attention_head import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    """
    Multiple parallel attention heads module
    """
    n_heads: int
    head_size: int
    n_embeddings: int
    block_size: int
    dropout_probability: float

    def __init__(
            self,
            n_heads: int, 
            head_size: int,
            n_embeddings: int,
            block_size: int,
            dropout_probability: float
        ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.head_size = head_size 
        self.block_size = block_size
        self.dropout_probability = dropout_probability

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                n_embeddings=self.n_embeddings,
                head_size=self.head_size,
                block_size=self.block_size,
                dropout_probability=self.dropout_probability
            ) for _ in range(self.n_heads)
        ])
        self.projection = nn.Sequential(
            nn.Linear(self.n_embeddings, self.n_embeddings),
            nn.Dropout(p=self.dropout_probability)
        )

    def forward(self, x: Tensor) -> Tensor:
        attention_heads_output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(attention_heads_output)
        return output
