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

    def __init__(
            self,
            n_heads: int, 
            n_embeddings: int,
            block_size: int
        ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size

        self.self_attention_heads = MultiHeadAttention(
            n_heads=self.n_heads,
            head_size=self.head_size,
            n_embeddings=self.n_embeddings,
            block_size=self.block_size
        )
        self.feed_forward = FeedForward(self.n_embeddings)

    @property
    def head_size(self) -> int:
        return self.n_embeddings // self.n_heads
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention_heads(x)
        x = x + self.feed_forward(x)
        return x
