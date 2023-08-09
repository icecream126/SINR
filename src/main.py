import os
import sys
sys.path.append('./')

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)

from src import datasets, models

dataset_dict = {
    'noaa': datasets.NOAA,
    'era5': datasets.ERA5,
    'sun360': datasets.SUN360
}
model_dict = {
    'inr': models.INR,
    'wire': models.WIRE,
    'shinr': models.SHINR,
    'swinr': models.SWINR
}

if __name__=='__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default='noaa', type=str)
    parser.add_argument("--model", default='inr', type=str)

    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--patience", default=5000, type=int)
    parser.add_argument('--spherical', default=False, action='store_true')
    parser.add_argument('--time', default=False, action='store_true')
    parser.add_argument('--in_memory', default=False, action='store_true')
    parser.add_argument('--panorama_idx',default=0, type=int)

    parser.add_argument("--wavelet_dim", type=int, default=64)
    parser.add_argument("--max_order", type=int, default=3)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument('--skip', default=True, action='store_true')
    parser.add_argument('--sine', default=False, action='store_true')
    parser.add_argument('--all_sine', default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_patience", type=int, default=1000)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    if args.dataset == 'sun360':
        train_dataset = dataset_dict[args.dataset](dataset_type='train', **vars(args))
        test_dataset = dataset_dict[args.dataset](dataset_type='test', **vars(args))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    else:
        train_dataset = dataset_dict[args.dataset](dataset_type='train', **vars(args))
        valid_dataset = dataset_dict[args.dataset](dataset_type='valid', **vars(args))
        test_dataset = dataset_dict[args.dataset](dataset_type='test', **vars(args))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    input_dim = (2 if args.spherical else 3) + (1 if args.time else 0)
    output_dim = train_dataset.target_dim
    model = model_dict[args.model](input_dim, output_dim, **vars(args))

    # Training
    if args.dataset != 'sun360':
        earlystopping_cb = EarlyStopping(monitor="valid_loss", patience=args.patience)
        checkpoint_cb = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            filename="best"
        )

    else:
        checkpoint_cb = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            filename="best"
        )
        
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        config=args, 
        project="SINR", 
        save_dir="experiment", 
        name=args.dataset+'/'+args.model+'/'+str(args.dataset_dir[8:])
    )

    logger.experiment.log(
        {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", None)}
    )

    if args.dataset != 'sun360':
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
        model.min_valid_loss = earlystopping_cb.best_score
        
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            max_epochs=300,
            log_every_n_steps=1,
            callbacks=[lrmonitor_cb, checkpoint_cb],
            logger=logger,
            gpus=torch.cuda.device_count(),
            strategy="ddp" if torch.cuda.device_count() > 1 else None
        )
        trainer.fit(model, train_loader)
        # model.min_train_loss = earlystopping_cb.best_score
        
    trainer.test(model, test_loader, 'best')