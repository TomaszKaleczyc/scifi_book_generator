import torch.nn as nn
from torch import Tensor

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class DecoderStack(nn.Module):
    """
    A single decoder stack
    """
    n_heads: int
    n_embeddings: int
    block_size: int
    dropout_probability: float

    def __init__(
            self,
            n_heads: int, 
            n_embeddings: int,
            block_size: int,
            dropout_probability: float
        ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.dropout_probability = dropout_probability

        self.self_attention_heads = MultiHeadAttention(
            n_heads=self.n_heads,
            head_size=self.head_size,
            n_embeddings=self.n_embeddings,
            block_size=self.block_size,
            dropout_probability=self.dropout_probability
        )
        self.feed_forward = FeedForward(self.n_embeddings, dropout_probability=self.dropout_probability)
        self.layer_norm1 = nn.LayerNorm(self.n_embeddings)
        self.layer_norm2 = nn.LayerNorm(self.n_embeddings)

    @property
    def head_size(self) -> int:
        return self.n_embeddings // self.n_heads
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention_heads(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
