# "args" : ["--dataset_dir","dataset/weather_time_gustsfc_cut/","--n_fourier 3", "--n_nodes_in_sample", "5000", "--lr", "0.001", "--n_layers", "8", "--skip","True", "--time", "True"]

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import sys
sys.path.append('./')

from src.data.spherical_dataset import GraphDataset
from src.models.spherical_inr import SHFeatINR

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--patience", default=5000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=0, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SHFeatINR.add_model_specific_args(parser)
    parser = GraphDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Data
    dataset = GraphDataset(**vars(args))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )

    # Model
    input_dim = dataset.n_fourier + (1 if dataset.time else 0)
    output_dim = dataset.target_dim
    model = SHFeatINR(input_dim, output_dim, len(dataset), **vars(args))

    # Training
    checkpoint_cb = ModelCheckpoint(
        monitor="loss", mode="min", save_last=True, filename="best"
    )
    earlystopping_cb = EarlyStopping(monitor="loss", patience=args.patience)
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        config=args, 
        project="SphericalINR", 
        save_dir="lightning_logs", 
        name='shfeat_org/'+str(args.dataset_dir[8:])+str(args.n_fourier)+'/'+str(args.n_nodes_in_sample)
        )
    logger.experiment.log(
        {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", None)}
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=5000,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
    )
    trainer.fit(model, loader)