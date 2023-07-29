import os
import sys
sys.path.append('./')

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)

from src.data.preprocessing.spherical_dataset import SphericalDataset
from src.data.preprocessing.era5_spherical_dataset import ERA5SphericalDataset
from src.models.spherical_inr import SphericalINR

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default='ERA5', type=str)
    parser.add_argument("--patience", default=5000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser = SphericalDataset.add_dataset_specific_args(parser)
    parser = SphericalINR.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Data
    if args.dataset=='ERA5':
        train_dataset = ERA5SphericalDataset(dataset_type='train',**vars(args))
        valid_dataset = ERA5SphericalDataset(dataset_type='valid',**vars(args)) # ERA5 compression task doesn't need validation
        test_dataset = ERA5SphericalDataset(dataset_type='test',**vars(args))
        need_time=False

    elif args.dataset=='NOAA': 
        train_dataset = SphericalDataset(dataset_type='train', **vars(args))
        valid_dataset = SphericalDataset(dataset_type='valid', **vars(args))
        test_dataset = SphericalDataset(dataset_type='test', **vars(args))
        need_time=True
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    # Model
    input_dim = 2 + (1 if train_dataset.time else 0)
    output_dim = train_dataset.target_dim
    model = SphericalINR(need_time, input_dim, output_dim, len(train_dataset), **vars(args))

    # Training
    earlystopping_cb = EarlyStopping(monitor="valid_loss", patience=args.patience)
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        monitor="valid_loss", 
        mode="min", 
        filename="best"
    )

    logger = WandbLogger(
        config=args, 
        project="SINR", 
        save_dir="experiment", 
        name='spherical/'+str(args.dataset_dir[8:])
        )
    logger.experiment.log(
        {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", None)}
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=-1,
        log_every_n_steps=1,
        callbacks=[earlystopping_cb, lrmonitor_cb, checkpoint_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None
    )


    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader, 'best')