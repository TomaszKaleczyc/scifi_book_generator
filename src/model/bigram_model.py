from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from .language_model import LanguageModel

import config


class BigramLanguageModel(LanguageModel, LightningModule):
    """
    Simple bigram model to generate new tokens
    """

    def __init__(self, vocabulary_size: int, learning_rate: float = config.DEFAULT_LEARNING_RATE) -> None:
        super(BigramLanguageModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.token_embedding_table = nn.Embedding(
            self.vocabulary_size, self.vocabulary_size
            )
        
    def forward(self, tokens: Tensor) -> Tensor:
        logits = self.token_embedding_table(tokens)
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
            logits = self(tokens)
            last_time_step_logits = logits[:, -1, :]
            probabilities = F.softmax(last_time_step_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
    