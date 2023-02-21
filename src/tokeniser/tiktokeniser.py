from typing import List

import tiktoken
import torch

from .tokeniser import Tokeniser


ENCODING_NAME = 'gpt2'


class TikTokeniser(Tokeniser):
    """
    Manages the tokenisation of data into character tokens
    """

    def __init__(self, text: str, encoding_name: str = ENCODING_NAME) -> None:
        super().__init__(text)
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.data = torch.tensor(self.encode(text))
        self.vocabulary = torch.unique(self.data)
        self.vocabulary_size = len(self.vocabulary)
        print('='*60)
        print('Set up tiktokeniser')
        print('Vocabulary size:', self.vocabulary_size)

    def encode(self, string: str) -> List[int]:
        return self.encoder.encode(string)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)
