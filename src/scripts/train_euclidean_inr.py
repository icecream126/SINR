import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data.euclidean_dataset import EuclideanDataset
from src.models.euclidean_inr import EuclideanINR

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument("--name", default="", type=str)
parser.add_argument("--patience", default=5000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_workers", default=0, type=int)
parser.add_argument("--plot_3d", action="store_true")
parser.add_argument("--plot_heat", action="store_true")
parser = pl.Trainer.add_argparse_args(parser)
parser = EuclideanINR.add_model_specific_args(parser)
parser = EuclideanDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

print(args)
# Data
dataset = EuclideanDataset(**vars(args))

loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
)

# Model
input_dim = 3 + (1 if dataset.time else 0)
output_dim = dataset.target_dim
model = EuclideanINR(input_dim, output_dim, len(dataset), **vars(args))

# Training
checkpoint_cb = ModelCheckpoint(
    monitor="loss", mode="min", save_last=True, filename="best"
)
earlystopping_cb = EarlyStopping(monitor="loss", patience=args.patience)
lrmonitor_cb = LearningRateMonitor(logging_interval="step")
logger = WandbLogger(
    project="SphericalINR", 
    save_dir="lightning_logs",
    name='euclidean/'+str(args.dataset_dir[8:])+'/'+str(args.n_nodes_in_sample)
    )
logger.experiment.log(
    {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", None)}
)
trainer = pl.Trainer.from_argparse_args(
    args,
    max_epochs=-1,
    log_every_n_steps=1,
    callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb],
    logger=logger,
    gpus=torch.cuda.device_count(),
    strategy="ddp" if torch.cuda.device_count() > 1 else None,
)
trainer.fit(model, loader)