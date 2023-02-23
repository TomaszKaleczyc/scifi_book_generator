import torch
import torch.nn as nn
from torch import Tensor

from .transformer_base import TransformerBase
from .self_attention_head import SelfAttentionHead

import config


class SingleHeadTransformer(TransformerBase):
    """
    Simplistic transformer implementation
    """

    def __init__(
            self, 
            vocabulary_size: int, 
            learning_rate: float = config.DEFAULT_LEARNING_RATE,
            block_size: int = config.DEFAULT_BLOCK_SIZE,
            n_embeddings: int = config.DEFAULT_N_EMBEDDINGS,
            n_heads: int = config.DEFAULT_N_HEADS
            ) -> None:
        super(SingleHeadTransformer, self).__init__(
            vocabulary_size=vocabulary_size,
            learning_rate=learning_rate,
            n_embeddings=n_embeddings,
            block_size=block_size,
            n_heads=1            
        )
        self.token_embedding_table = nn.Embedding(
            self.vocabulary_size, self.n_embeddings
            )
        self.position_embedding_table = nn.Embedding(
            self.block_size, self.n_embeddings
        )
        self.self_attention_head = SelfAttentionHead(
            n_embedding=self.n_embeddings,
            head_size=self.head_size,
            block_size=self.block_size
        )
        self.lm_head = nn.Linear(self.head_size, self.vocabulary_size)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        token_embeddings = self.token_embedding_table(tokens)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        input_embeddings = token_embeddings + position_embeddings
        output_embeddings = self.self_attention_head(input_embeddings)
        logits = self.lm_head(output_embeddings)
        return logits
