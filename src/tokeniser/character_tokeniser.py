from typing import List

from .tokeniser import Tokeniser


class CharacterTokeniser(Tokeniser):
    """
    Manages the tokenisation of data into character tokens
    """
    character_to_token: dict
    token_to_character: dict
    characters: List[str]
    vocabulary_size: int
    
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.characters = list(set(self.text))
        self.vocabulary_size = len(self.characters)
        self.character_to_token = {character: idx  for idx, character in enumerate(self.characters)}
        self.token_to_character = {idx: character for idx, character in enumerate(self.characters)}
        print('='*60)
        print('Set up character tokeniser')
        print('Vocabulary size:', self.vocabulary_size)
        print('Vocabulary:', self.vocabulary)

    @property
    def vocabulary(self) -> str:
        return ''.join(self.characters)

    def encode(self, string: str) -> List[int]:
        return [self.character_to_token[character] for character in string]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.token_to_character[token] for token in tokens])
