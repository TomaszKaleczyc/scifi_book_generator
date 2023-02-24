import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import LMDataModule
from model import TRANSFORMERS


MODEL_NAME = 'multi_head_transformer'
TOKENISER = 'character'
BLOCK_SIZE = 8
BATCH_SIZE = 8
VALIDATION_SET_RATIO = 0.1
LEARNING_RATE = 1e-3
N_HEADS = 4
N_EMBEDDINGS = 32
LIMIT_TRAIN_BATCHES_RATIO = 3e-4
LIMIT_VAL_BATCHES_RATIO = LIMIT_TRAIN_BATCHES_RATIO

NUM_EPOCHS = 1
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
    block_size=BLOCK_SIZE,
    n_embeddings=N_EMBEDDINGS,
    n_heads=N_HEADS
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


trainer.fit(
    model,
    train_dataloaders=data_module.train_dataloader(),
    val_dataloaders=data_module.val_dataloader()
)


inference_input = torch.zeros((1, 1), dtype=torch.long).to(model.device)
max_new_tokens = 1000
inference_results = model.generate(inference_input, max_new_tokens)
print()
print('Inference test:')
for inference_result in inference_results:
    print(data_module.decode(inference_result))
