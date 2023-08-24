from argparse import ArgumentParser

import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.utils import mse2psnr
from utils.visualize import visualize

from datasets import spatial, temporal
from models import relu, siren, wire, shinr, swinr, shiren

model_dict = {
    'relu': relu,
    'siren': siren,
    'wire': wire,
    'shinr': shinr,
    'swinr': swinr,
    'shiren': shiren,
}

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model", type=str, default='swinr')

    # Dataset argument
    parser.add_argument("--panorama_idx", type=int, default=0)
    parser.add_argument("--normalize", default=False, action='store_true')

    # Model argument
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument('--skip', default=False, action='store_true')
    parser.add_argument("--omega", type=float, default=1.)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--levels", type=int, default=4)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_patience", type=int, default=1000)

    parser.add_argument("--plot", default=False, action='store_true')
    args = parser.parse_args()

    pl.seed_everything(0)
    
    args.time = True if 'temporal' in args.dataset_dir else False
    args.input_dim = 3 + (1 if args.time else 0)
    args.output_dim = 3 if 'sun360' in args.dataset_dir else 1

    # Log
    logger = WandbLogger(
        config=args,
        name=args.model,
    )

    # Dataset
    dataset = temporal if args.time else spatial

    train_dataset = dataset.Dataset(dataset_type='train', **vars(args))
    valid_dataset = dataset.Dataset(dataset_type='valid', **vars(args))
    test_dataset = dataset.Dataset(dataset_type='test', **vars(args))
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    # Model
    model = model_dict[args.model].INR(**vars(args))

    # Learning
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        monitor="valid_loss",
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
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
    )

    trainer.fit(model, train_loader, valid_loader)
    res = trainer.test(model, test_loader, 'best')[0]

    wandb.log({
        "test_rmse": np.sqrt(res['test_mse']),
        "test_psnr": mse2psnr(res['test_mse']),
        "best_valid_loss": checkpoint_cb.best_model_score.item(),
        "best_valid_psnr": mse2psnr(checkpoint_cb.best_model_score.item()),
    })

    if args.plot and 'spatial' in args.dataset_dir:
        dataset = dataset.Dataset(dataset_type='all', **vars(args))
        visualize(dataset, model, args, 'HR')
        visualize(train_dataset, model, args, 'LR')