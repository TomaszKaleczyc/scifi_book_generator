import torch
import torch.nn as nn
from torch import Tensor

from .transformer_base import TransformerBase
from .transformer_modules import DecoderStack

import config


class FinalMultiHeadTransformer(TransformerBase):
    """
    Multi-head transformer implementation as per the paper
    """
    n_stacks: int

    def __init__(
            self, 
            vocabulary_size: int, 
            learning_rate: float = config.DEFAULT_LEARNING_RATE,
            block_size: int = config.DEFAULT_BLOCK_SIZE,
            n_embeddings: int = config.DEFAULT_N_EMBEDDINGS,
            n_heads: int = config.DEFAULT_N_HEADS,
            n_stacks: int = config.DEFAULT_N_STACKS,
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
        self.n_stacks = n_stacks
        self.token_embedding_table = nn.Embedding(
            self.vocabulary_size, self.n_embeddings
            )
        self.position_embedding_table = nn.Embedding(
            self.block_size, self.n_embeddings
        )
        stack_args = dict(
            n_heads=self.n_heads,
            n_embeddings=self.n_embeddings,
            block_size=self.block_size,
            dropout_probability=self.dropout_probability            
        )
        stack_list = [DecoderStack(**stack_args) for _ in range(self.n_stacks)]
        self.stacks = nn.Sequential(*stack_list)
        self.lm_head = nn.Linear(self.n_embeddings, self.vocabulary_size)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        token_embeddings = self.token_embedding_table(tokens)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        input_embeddings = token_embeddings + position_embeddings
        output_embeddings = self.stacks(input_embeddings)
        logits = self.lm_head(output_embeddings)
        return logits
