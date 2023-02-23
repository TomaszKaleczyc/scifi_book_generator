from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import LMDataModule
from model import Transformer as Model

import config

MODEL_NAME = 'bigram'
TOKENISER = 'character' #'tiktoken'
BLOCK_SIZE = 8
BATCH_SIZE = 8
N_EMBEDDING = 32
VALIDATION_SET_RATIO = 0.1
LEARNING_RATE = 1e-3

NUM_EPOCHS=1
SAVE_DIR = '../output'

data_module = LMDataModule(
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    validation_set_ratio=VALIDATION_SET_RATIO,
    tokeniser=TOKENISER
)

model = Model(
    vocabulary_size=data_module.vocabulary_size,
    learning_rate=LEARNING_RATE,
    block_size=data_module.block_size,
    n_embedding=N_EMBEDDING
    )

tokens, targets = next(iter(data_module.train_dataloader()))
print("Input tokens:", tokens.shape)
print(tokens[0])

logits = model(tokens)
print("Logits:", logits.shape)
print(logits[0])
