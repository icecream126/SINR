from argparse import ArgumentParser

import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.utils import mse2psnr, calculate_ssim
from utils.visualize import visualize, visualize_denoising
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"


from datasets import denoising
from models import relu, siren, wire, shinr, swinr, shiren

model_dict = {
    "relu": relu,
    "siren": siren,
    "wire": wire,
    "shinr": shinr,
    "swinr": swinr,
    "shiren": shiren,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model", type=str, default="swinr")

    # Dataset argument
    parser.add_argument("--panorama_idx", type=int, default=0)
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--tau", type=float, default=3e1)
    parser.add_argument("--snr", type=float, default=2)

    # Model argument
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--skip", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--posenc_freq", type=int, default=10)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_patience", type=int, default=1000)

    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()

    pl.seed_everything(0)

    args.time = True if "temporal" in args.dataset_dir else False
    args.input_dim = 3 + (1 if args.time else 0)
    args.output_dim = (
        3 if "360" in args.dataset_dir else 1
    )  # sun360, flickr360 takes 3 else 1

    # Log
    logger = WandbLogger(
        config=args,
        name=args.model,
        project="denoising",
        # mode='disabled',
    )

    # Dataset
    dataset = denoising

    train_dataset = dataset.Dataset(**vars(args))
    test_dataset = dataset.Dataset(**vars(args))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    model = model_dict[args.model].DENOISING_INR(**vars(args))

    # Learning
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss_orig", mode="min", filename="best"
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

    trainer.fit(model, train_loader)
    res = trainer.test(model, test_loader, "best")[0]

    logger.experiment.log(
        {
            "test_rmse": np.sqrt(res["test_mse"]),
            "test_psnr": mse2psnr(res["test_mse"]),
            "best_orig_loss": checkpoint_cb.best_model_score.item(),
            "best_orig_psnr": mse2psnr(checkpoint_cb.best_model_score.item()),
        }
    )

    dataset_all = dataset.Dataset(**vars(args))
    logger.experiment.log(
        {"test_ssim": calculate_ssim(model, dataset_all, args.output_dim)}
    )

    if args.plot:
        dataset_all = denoising.Dataset(**vars(args))
        visualize_denoising(dataset_all, model, args, logger=logger)
