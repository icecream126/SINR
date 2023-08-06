import os
import sys
sys.path.append('./')

from argparse import ArgumentParser

import wandb
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)


from src.data.preprocessing.era5_spherical_dataset import ERA5SphericalDataset
from src.models.spherical_inr import SphericalINR

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default='ERA5', type=str)
    parser.add_argument("--patience", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser = ERA5SphericalDataset.add_dataset_specific_args(parser)
    parser = SphericalINR.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    with wandb.init(config = vars(args)):

        with open('./sweeps/yaml/era5_spherical.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        wandb.config.update(config)
        args = wandb.config
            
        # Data
        train_dataset = ERA5SphericalDataset(dataset_type='train',**dict(wandb.config))
        valid_dataset = ERA5SphericalDataset(dataset_type='valid', **dict(wandb.config))
        test_dataset = ERA5SphericalDataset(dataset_type='test',**dict(wandb.config))
        need_time=False
        
        train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=wandb.config.n_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=wandb.config.n_workers)
        test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=wandb.config.n_workers)

        # Model
        input_dim = 2 + (1 if train_dataset.time else 0)
        output_dim = train_dataset.target_dim
        model = SphericalINR(need_time, input_dim, output_dim, len(train_dataset), **dict(wandb.config))

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