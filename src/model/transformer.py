from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from .language_model import LanguageModel
from .self_attention_head import SelfAttentionHead

import config


class Transformer(LanguageModel, LightningModule):
    """
    Transformer implementation
    """
    n_embeddings: int
    block_size: int
    n_heads: int

    def __init__(
            self, 
            vocabulary_size: int, 
            learning_rate: float = config.DEFAULT_LEARNING_RATE,
            block_size: int = config.DEFAULT_BLOCK_SIZE,
            n_embeddings: int = config.DEFAULT_N_EMBEDDINGS,
            n_heads: int = config.DEFAULT_N_HEADS
            ) -> None:
        super(Transformer, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.n_heads = n_heads

        # model structure:
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

    @property
    def head_size(self) -> int:
        return self.n_embeddings // self.n_heads
        
    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        token_embeddings = self.token_embedding_table(tokens)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        input_embeddings = token_embeddings + position_embeddings
        output_embeddings = self.self_attention_head(input_embeddings)
        logits = self.lm_head(output_embeddings)
        return logits
    
    def _loss_step(
            self, 
            batch: Tuple[Tensor, Tensor], 
            batch_idx: int,
            dataset_name: str,
            criterion: Callable = F.cross_entropy
            ) -> Tensor:
        tokens, targets = batch
        logits = self(tokens)
        B, T, C = logits.shape
        loss = criterion(
            logits.view(B*T, C), 
            targets.view(B*T)
            )
        self.log(f'{dataset_name}/loss', loss)
        return loss
    
    def generate(self, tokens: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            context_tokens = tokens[:, -self.block_size:]
            logits = self(context_tokens)
            last_time_step_logits = logits[:, -1, :]
            probabilities = F.softmax(last_time_step_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
