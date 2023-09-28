from argparse import ArgumentParser

import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.visualize import visualize_era5, visualize_360, visualize_synthetic
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

from datasets import spatial, temporal, temporal_ginr, spatial_ginr
from models import relu, siren, wire, shinr, swinr, shiren, ginr, swinr_adap_all, swinr_adap_omega, swinr_adap_sigma, gauss,gauss_act

model_dict = {
    "relu": relu,
    "siren": siren,
    "wire": wire,
    "shinr": shinr,
    "swinr": swinr,
    "swinr_adap_all": swinr_adap_all,
    "swinr_adap_omega": swinr_adap_omega,
    "swinr_adap_sigma": swinr_adap_sigma,
    "shiren": shiren,
    "ginr": ginr,
    "gauss": gauss,
    "gauss_act":gauss_act,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model", type=str, default="swinr")

    # Dataset argument
    parser.add_argument("--panorama_idx", type=int, default=2)
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--zscore_normalize", default=False, action="store_true")
    parser.add_argument("--data_year", default=None)  # For weather temporal
    parser.add_argument("--time_resolution", type=int, default=24)

    # Model argument
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--skip", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--posenc_freq", type=int, default=10)
    parser.add_argument("--relu", default=False, action="store_true")
    parser.add_argument("--gauss_scale", type=int, default=10)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_patience", type=int, default=1000)
    
    # Ablation argument
    parser.add_argument("--learn_rotate", default=True, action="store_false")
    parser.add_argument("--learn_dilate", default=True, action="store_false")

    # GINR argument
    parser.add_argument("--n_fourier", type=int, default=34)

    parser.add_argument("--project_name", type=str, default="fair_superres")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.time = True if "temporal" in args.dataset_dir else False
    args.input_dim = 3 + (1 if args.time else 0)
    args.output_dim = (
        3 if "360" in args.dataset_dir else 1
    )  # sun360, flickr360 takes 3 else 1

    # Log
    logger = WandbLogger(config=args, name=args.model, project=args.project_name)

    # Dataset
    if args.time:
        dataset = temporal
        if args.model == "ginr":
            dataset = temporal_ginr
    else:
        dataset = spatial
        if args.model == "ginr":
            dataset = spatial_ginr

    train_dataset = dataset.Dataset(dataset_type="train", **vars(args))
    test_dataset = dataset.Dataset(dataset_type="test", **vars(args))
    if args.model == "ginr":
        args.input_dim = train_dataset.n_fourier + (1 if args.time else 0)

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
    model = model_dict[args.model].INR(**vars(args))

    # Pass scaler from dataset to model
    model.scaler = train_dataset.scaler
    model.normalize = train_dataset.zscore_normalize or train_dataset.normalize

    # Learning
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        monitor="batch_train_mse", mode="min", filename="best"
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
    trainer.test(model, test_loader, "best")

    if args.plot:
        dataset_all = dataset.Dataset(dataset_type="all", **vars(args))

        if "era5" in args.dataset_dir:
            # Currently only HR visualized
            visualize_era5('all',
                dataset_all, model, args.dataset_dir + "/data.nc", logger, args
            )
            # TODO : Visualize LR
            # Visualizing LR with earthmap is tricky due to interpolation
            # visualize_era5_LR(train_dataset, model, args.dataset_dir+"/data.nc", logger, args)
            # Maybe we don't need LR visualizations. At most, we would need the ground truth.
            # i.e., we don't need error map, prediction of LR visualizations
        elif '360' in args.dataset_dir:
            visualize_360('all',dataset_all, model, args, "HR", logger=logger)
            visualize_360('train',train_dataset, model, args, "LR", logger=logger)
        else:
            visualize_synthetic('all', dataset_all, model, args,"HR", logger=logger)
            visualize_synthetic('train', train_dataset, model, args,"LR", logger=logger)
            
