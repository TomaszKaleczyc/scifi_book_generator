import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import LMDataModule
from model import BigramLanguageModel, TRANSFORMERS

import config


torch.manual_seed(config.RANDOM_SEED)

# MODEL_NAME = 'bigram'
# MODEL_NAME = 'single_head_transformer'
# MODEL_NAME = 'multi_head_transformer'
MODEL_NAME = 'final_multi_head_transformer'

TOKENISER = 'character'
# BLOCK_SIZE = 256
BLOCK_SIZE = 384
BATCH_SIZE = 32
VALIDATION_SET_RATIO = 0.1
# LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4
# N_STACKS = 6
N_STACKS = 8
N_HEADS = 8
# N_STACKS = 12
N_EMBEDDINGS = 384
DROPOUT_PROBABILITY = 0.2

LIMIT_TRAIN_BATCHES_RATIO = 3e-4 * (BATCH_SIZE // 8)
LIMIT_VAL_BATCHES_RATIO = LIMIT_TRAIN_BATCHES_RATIO

NUM_EPOCHS = 1
SAVE_DIR = '../output'


data_module = LMDataModule(
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    validation_set_ratio=VALIDATION_SET_RATIO,
    tokeniser=TOKENISER
)


if MODEL_NAME == 'bigram':
    model = BigramLanguageModel(
        vocabulary_size=data_module.vocabulary_size,
        learning_rate=LEARNING_RATE
    )
else:
    model = TRANSFORMERS[MODEL_NAME](
        vocabulary_size=data_module.vocabulary_size,
        learning_rate=LEARNING_RATE,
        block_size=BLOCK_SIZE,
        n_embeddings=N_EMBEDDINGS,
        n_heads=N_HEADS,
        n_stacks=N_STACKS,
        dropout_probability=DROPOUT_PROBABILITY
    )


callbacks = [
    ModelCheckpoint(
        filename=MODEL_NAME+'{epoch}-{validation/loss:.3f}',
        monitor='validation/loss',
        verbose=True,
        save_top_k=3,
        mode='min'
    )
]


trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    fast_dev_run=False,
    default_root_dir=SAVE_DIR,
    limit_train_batches=LIMIT_TRAIN_BATCHES_RATIO,
    limit_val_batches=LIMIT_VAL_BATCHES_RATIO,
    accelerator='gpu',
    devices=1,
    callbacks=callbacks
)

print('='*60)
print('MODEL TRAINING:')
trainer.fit(
    model,
    train_dataloaders=data_module.train_dataloader(),
    val_dataloaders=data_module.val_dataloader()
)


inference_input = torch.zeros((1, 1), dtype=torch.long).to(model.device)
max_new_tokens = 1000
inference_results = model.generate(inference_input, max_new_tokens)
print('='*60)
print('INFERENCE TEST:')
for inference_result in inference_results:
    print(data_module.decode(inference_result))
