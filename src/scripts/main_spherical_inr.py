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
from src.models.spherical_inr import SphericalINR

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--patience", default=5000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SphericalDataset.add_dataset_specific_args(parser)
    parser = SphericalINR.add_model_specific_args(parser)
    args = parser.parse_args()

    # Data
    args.dataset_type = 'train'
    train_dataset = SphericalDataset(**vars(args))
    args.dataset_type = 'valid'
    valid_dataset = SphericalDataset(**vars(args))
    args.dataset_type = 'test'
    test_dataset = SphericalDataset(**vars(args))
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
    )

    # Model
    input_dim = 2 + (1 if train_dataset.time else 0)
    output_dim = train_dataset.target_dim
    model = SphericalINR(input_dim, output_dim, len(train_dataset), **vars(args))

    # Training
    checkpoint_cb = ModelCheckpoint(
        monitor="valid_loss", mode="min", save_last=True, filename="best"
    )
    earlystopping_cb = EarlyStopping(monitor="valid_loss", patience=args.patience)
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        config=args, 
        project="SphericalINR", 
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
        callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader, 'best')