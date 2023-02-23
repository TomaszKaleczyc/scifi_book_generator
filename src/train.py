
from dataset import LMDataModule
from model import TRANSFORMERS


MODEL_NAME = 'multi_head_transformer'
TOKENISER = 'character' #'tiktoken'
BLOCK_SIZE = 8
BATCH_SIZE = 8
N_EMBEDDINGS = 32
VALIDATION_SET_RATIO = 0.1
LEARNING_RATE = 1e-3
N_HEADS = 4

NUM_EPOCHS=1
SAVE_DIR = '../output'

data_module = LMDataModule(
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    validation_set_ratio=VALIDATION_SET_RATIO,
    tokeniser=TOKENISER
)

model = TRANSFORMERS[MODEL_NAME](
    vocabulary_size=data_module.vocabulary_size,
    learning_rate=LEARNING_RATE,
    block_size=data_module.block_size,
    n_embeddings=N_EMBEDDINGS,
    n_heads=N_HEADS,
    )

tokens, targets = next(iter(data_module.train_dataloader()))
print("Input tokens:", tokens.shape)
print(tokens[0])

logits = model(tokens)
print("Logits:", logits.shape)
print(logits[0])
