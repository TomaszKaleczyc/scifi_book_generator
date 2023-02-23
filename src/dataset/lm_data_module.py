from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import tokeniser as tknsr
from .lm_dataset import LMDataset

import config


class LMDataModule(LightningDataModule):
    """
    Manages the language model model dataset
    """
    block_size: int
    raw_data_path: str
    batch_size: int
    validation_set_ratio: float
    tokeniser_class: tknsr.Tokeniser
    tokeniser: tknsr.Tokeniser
    train_dataset: LMDataset
    val_dataset: LMDataset

    @property
    def vocabulary_size(self) -> int:
        return self.tokeniser.vocabulary_size

    def __init__(
            self, 
            block_size: int = config.DEFAULT_BLOCK_SIZE,
            raw_data_path: str = config.RAW_DATA_PATH,
            validation_set_ratio: float = config.DEFAULT_VALIDATION_SET_RATIO,
            batch_size: int = config.DEFAULT_BATCH_SIZE,
            tokeniser: str = config.DEFAULT_TOKENISER
            ) -> None:
        self.block_size = block_size
        self.raw_data_path = raw_data_path
        self.batch_size = batch_size
        self.tokeniser_class = self._get_tokeniser_class(tokeniser)
        self.validation_set_ratio = validation_set_ratio
        self._set_datasets()

    def _get_tokeniser_class(self, tokeniser: str) -> tknsr.Tokeniser:
        """
        Returns the selected tokeniser class
        """
        tokenisers = {
            'character': tknsr.CharacterTokeniser,
            'tiktokeniser': tknsr.TikTokeniser,
        }
        selected_tokeniser = tokenisers.get(tokeniser)
        if selected_tokeniser:
            return selected_tokeniser
        raise NotImplementedError

    def _set_datasets(self) -> None:
        """
        Creates the Dataset objects
        """
        with open(self.raw_data_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f'Total corpus length: {len(text)}')
        self.tokeniser = self.tokeniser_class(text)
        train_ratio = 1 - self.validation_set_ratio
        validation_cutoff = int(len(text) * train_ratio)
        train_text = text[:validation_cutoff]
        self.train_dataset = LMDataset(train_text, self.block_size, self.tokeniser)
        val_text = text[validation_cutoff:]
        self.val_dataset = LMDataset(val_text, self.block_size, self.tokeniser)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=config.DEFAULT_NUM_WORKERS
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=config.DEFAULT_NUM_WORKERS
            )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def decode(self, tokens: Tensor) -> str:
        return self.train_dataset.decode(tokens)
