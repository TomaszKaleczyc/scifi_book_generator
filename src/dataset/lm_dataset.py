import torch
from torch import Tensor
from torch.utils.data import Dataset

from tokeniser import Tokeniser


DEFAULT_DATA_TYPE = torch.long


class LMDataset(Dataset):
    """
    Manages creating a language model dataset
    """
    block_size: int
    tokerniser: Tokeniser
    encoded_data: Tensor

    def __init__(self, text: str, block_size: int, tokeniser: Tokeniser, data_type: torch.dtype = DEFAULT_DATA_TYPE) -> None:
        self.block_size = block_size
        self.data_type = data_type
        self.tokerniser = tokeniser
        self.encoded_data = torch.tensor(self.tokerniser.encode(text), dtype=self.data_type)
        print("="*60)
        print(f"Imported data of shape {self.encoded_data.shape} and type {self.encoded_data.dtype}")

    def __len__(self) -> int:
        return len(self.encoded_data) - (self.block_size + 1)
    
    def __getitem__(self, idx: int) -> Tensor:
        tokens = self.encoded_data[idx: idx + self.block_size + 1]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        return input_tokens, target_tokens
    
    def decode(self, token_sequence: Tensor) -> str:
        tokens = token_sequence.tolist()
        return self.tokerniser.decode(tokens)
