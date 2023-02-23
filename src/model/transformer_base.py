from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from .language_model import LanguageModel


class TransformerBase(LanguageModel, LightningModule):
    """
    Transformer base implementation
    """
    n_embeddings: int
    block_size: int
    n_heads: int

    def __init__(
            self, 
            vocabulary_size: int, 
            learning_rate: float,
            block_size: int,
            n_embeddings: int,
            n_heads: int
            ) -> None:
        super(TransformerBase, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.n_heads = n_heads

    @property
    def head_size(self) -> int:
        return self.n_embeddings // self.n_heads
           
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
