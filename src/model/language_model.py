from abc import ABC, abstractmethod
from typing import Callable, Tuple

from torch import optim, Tensor


class LanguageModel(ABC):
    """
    Language model abstraction
    """
    vocabulary_size: int
    learning_rate: float

    @abstractmethod
    def generate(self, tokens: Tensor, max_new_tokens: int) -> Tensor:
        """
        Based on imput generates a given number of tokens
        """

    @abstractmethod
    def _loss_step(
            self, 
            batch: Tuple[Tensor, Tensor], 
            batch_idx: int,
            dataset_name: str,
            criterion: Callable
            ) -> Tensor:
        """
        Basic loss generating step for model training
        """
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._loss_step(batch, batch_idx, dataset_name='train')    
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._loss_step(batch, batch_idx, dataset_name='validation')
    
    def configure_optimizers(self):
        """
        Configuring the net optimization methods
        """
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
