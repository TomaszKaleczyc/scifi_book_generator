import torch
import torch.nn as nn
from torch import Tensor

from .transformer_base import TransformerBase
from .transformer_modules import FeedForward, MultiHeadAttention

import config


class MultiHeadTransformer(TransformerBase):
    """
    Multi-head transformer implementation
    """

    def __init__(
            self, 
            vocabulary_size: int, 
            learning_rate: float = config.DEFAULT_LEARNING_RATE,
            block_size: int = config.DEFAULT_BLOCK_SIZE,
            n_embeddings: int = config.DEFAULT_N_EMBEDDINGS,
            n_heads: int = config.DEFAULT_N_HEADS,
            n_stacks: int = 1,
            dropout_probability: float = config.DEFAULT_DROPOUT_PROBABILITY            
        ) -> None:
        super().__init__(
            vocabulary_size=vocabulary_size,
            learning_rate=learning_rate,
            n_embeddings=n_embeddings,
            block_size=block_size,
            n_heads=n_heads,
            dropout_probability=dropout_probability            
        )
        self.token_embedding_table = nn.Embedding(
            self.vocabulary_size, self.n_embeddings
            )
        self.position_embedding_table = nn.Embedding(
            self.block_size, self.n_embeddings
        )
        self.self_attention_heads = MultiHeadAttention(
            n_heads=self.n_heads,
            n_embeddings=self.n_embeddings,
            head_size=self.head_size,
            block_size=self.block_size,
            dropout_probability=dropout_probability
        )
        self.feed_forward = FeedForward(self.n_embeddings, dropout_probability=self.dropout_probability)
        self.lm_head = nn.Linear(self.n_embeddings, self.vocabulary_size)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        token_embeddings = self.token_embedding_table(tokens)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        input_embeddings = token_embeddings + position_embeddings
        attention_embeddings = self.self_attention_heads(input_embeddings)
        output_embeddings = self.feed_forward(attention_embeddings)
        logits = self.lm_head(output_embeddings)
        return logits
