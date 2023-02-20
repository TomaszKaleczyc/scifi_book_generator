from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .character_dataset import CharacterDataset, CharacterTokeniser

import config


class CharacterDataModule(LightningDataModule):
    """
    Manages the model dataset
    """
    block_size: int
    raw_data_path: str
    batch_size: int
    validation_set_ratio: float
    tokeniser: CharacterTokeniser
    train_dataset: CharacterDataset
    val_dataset: CharacterDataset

    def __init__(
            self, 
            block_size: int = config.BLOCK_SIZE,
            raw_data_path: str = config.RAW_DATA_PATH,
            validation_set_ratio: float = config.VALIDATION_SET_RATIO,
            batch_size: int = config.BATCH_SIZE
            ) -> None:
        self.block_size = block_size
        self.raw_data_path = raw_data_path
        self.batch_size = batch_size
        self.validation_set_ratio = validation_set_ratio
        self._set_datasets()

    def _set_datasets(self) -> None:
        """
        Creates the Dataset objects
        """
        with open(self.raw_data_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f'Total corpus length: {len(text)}')
        self.tokeniser = CharacterTokeniser(text)
        train_ratio = 1 - self.validation_set_ratio
        validation_cutoff = int(len(text) * train_ratio)
        train_text = text[:validation_cutoff]
        self.train_dataset = CharacterDataset(train_text, self.block_size, self.tokeniser)
        val_text = text[validation_cutoff:]
        self.val_dataset = CharacterDataset(val_text, self.block_size, self.tokeniser)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=config.NUM_WORKERS
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS
            )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
