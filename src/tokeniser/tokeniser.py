from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class Tokeniser(ABC):
    """
    Abstract class representing tokenisers
    """
    vocabulary_size: int
    vocabulary: str
    text: str
    data: Tensor

    def __init__(self, text: str) -> None:
        self.text = text

    @abstractmethod
    def encode(self, string: str) -> List[int]:
        """
        Encodes given string into tokens
        """

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decodes given tokens to string
        """
