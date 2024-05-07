import motorbike_project as mp
import torch.nn as nn
import torch
import pytorch_lightning as pl
import pandas as pd

model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
model = LitModule.load_from_checkpoint(
          checkpoint_path=hparams.model.checkpoint_path,
          **hparams.model # nested dict with model hyperparameters
  )

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)