from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets import noaa, era5, sun360, circle
from model import INR

dataset_dict = {
    'dpt2m': noaa.NOAA,
    'gustsfc': noaa.NOAA,
    'tcdcclm': noaa.NOAA,
    'era5_spa': era5.ERA5,
    'era5_temp': era5.ERA5,
    'sun360': sun360.SUN360,
    'circle': circle.CIRCLE,
}

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='relu')
    parser.add_argument("--dataset", type=str, default='dpt2m')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_ratio", type=float, default=0.05)
    parser.add_argument("--time_scale", type=float, default=60.)
    parser.add_argument("--z_scale", type=float, default=100.)
    parser.add_argument('--panorama_idx', type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--max_order", type=int, default=3)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument('--skip', default=False, action='store_true')
    parser.add_argument("--omega", type=float, default=20.)
    parser.add_argument("--sigma", type=float, default=10.)

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_patience", type=int, default=500)
    parser.add_argument("--plot", default=False, action='store_true')
    args = parser.parse_args()
    
    args.validation = True # False if args.dataset == 'sun360' else True
    args.spherical = True if args.model == 'shinr' else False
    args.time = False if args.dataset in ['sun360', 'circle'] else True

    logger = WandbLogger(
        config=args,
        project="SINR",
        name=f'{args.dataset}/{args.model}/SR'
    )

    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        monitor="valid_loss" if args.validation else "train_loss",
        mode="min",
        filename="best"
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[lrmonitor_cb, checkpoint_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None
    )

    # Data
    train_dataset = dataset_dict[args.dataset](dataset_type='train', **vars(args))
    valid_dataset = dataset_dict[args.dataset](dataset_type='valid', **vars(args))
    test_dataset = dataset_dict[args.dataset](dataset_type='test', **vars(args))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    pl.seed_everything(1234)
    
    # Model
    args.input_dim = (2 if args.spherical else 3) + (1 if args.time else 0)
    args.output_dim = 3 if args.dataset=='sun360' else 1
    model = INR(**vars(args))

    # Learning
    if args.validation:
        trainer.fit(model, train_loader, valid_loader)
        model.min_valid_loss = checkpoint_cb.best_model_score
    else:
        trainer.fit(model, train_loader)
    trainer.test(model, test_loader, 'best')