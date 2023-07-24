import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import sys
sys.path.append('./')

from src.data.preprocessing import euclidean_dataset, graph_dataset, spherical_dataset
from src.models import euclidean_inr, graph_inr, spherical_inr

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--model", default="spherical", type=str)
    parser.add_argument("--plot_3d", action="store_true")
    parser.add_argument("--plot_heat", action="store_true")
    parser.add_argument("--patience", default=5000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=0, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.model == 'euclidean':
        parser = euclidean_dataset.Dataset.add_dataset_specific_args(parser)
        parser = euclidean_inr.EuclideanINR.add_model_specific_args(parser)
    elif args.model == 'graph':
        parser = graph_dataset.Dataset.add_dataset_specific_args(parser)
        parser = graph_inr.GraphINR.add_model_specific_args(parser)
    elif args.model == 'spherical':
        parser = spherical_dataset.Dataset.add_dataset_specific_args(parser)
        parser = spherical_inr.SphericalINR.add_model_specific_args(parser)
    args = parser.parse_args()

    # Data
    args.dataset_type = 'train'
    train_dataset = Dataset(**vars(args))
    args.dataset_type = 'valid'
    valid_dataset = Dataset(**vars(args))

    # Model
    input_dim = train_dataset.n_fourier + (1 if train_dataset.time else 0)
    output_dim = train_dataset.target_dim
    model = SHFeatINR(input_dim, output_dim, len(train_dataset), **vars(args))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    
    # Training
    checkpoint_cb = ModelCheckpoint(
        monitor="valid_loss", mode="min", save_last=True, filename="best"
    )
    earlystopping_cb = EarlyStopping(monitor="valid_loss", patience=args.patience)
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        config=args,
        project="SphericalINR", 
        save_dir="lightning_logs", 
        name=args.model+'/'+str(args.dataset_dir[8:])+str(args.n_fourier)+'/'+str(args.n_nodes_in_sample)
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
    trainer.fit(model, train_loader, valid_loader)
