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
from models import coolchic_interp, ngp_interp, learnable, relu, siren, wire, shinr, swinr, shiren, ginr, swinr_learn_all, swinr_adap_all, swinr_adap_omega, swinr_adap_sigma, gauss,gauss_act, ewinr, swinr_pe

model_dict = {
    "relu": relu,
    "siren": siren,
    "wire": wire,
    "shinr": shinr,
    "swinr": swinr,
    "swinr_pe": swinr_pe,
    "swinr_learn_all": swinr_learn_all,
    "swinr_adap_all": swinr_adap_all,
    "swinr_adap_omega": swinr_adap_omega,
    "swinr_adap_sigma": swinr_adap_sigma,
    "shiren": shiren,
    "ginr": ginr,
    "gauss": gauss,
    "gauss_act":gauss_act,
    "ewinr":ewinr,
    "learnable":learnable,
    "coolchic_interp":coolchic_interp,
    "ngp_interp":ngp_interp,
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
    parser.add_argument("--downscale_factor", type=int, default=2)

    # Model argument
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--skip", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--posenc_freq", type=int, default=10)
    parser.add_argument("--relu", default=False, action="store_true")
    parser.add_argument("--gauss_scale", type=float, default=10.0)
    parser.add_argument("--omega_0", default=10.0, type=float)
    parser.add_argument("--sigma_0", default=10.0, type=float)
    parser.add_argument("--tau_0", default=10.0, type=float)
    parser.add_argument("--wavelet_dim", default=1024, type=int)
    parser.add_argument("--freq_enc_type", default='sin', type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--mapping_size", default=256, type=int)
    parser.add_argument("--level", default=2, type=int)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_patience", type=int, default=1000)
    parser.add_argument("--task",type=str, default='sr', choices=['reg','sr']) # regression and super-resolution
    
    # Ablation argument
    parser.add_argument("--stop_rotate", default=True, action="store_false")
    parser.add_argument("--stop_dilate", default=True, action="store_false")

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
    logger = WandbLogger(config=args, name=args.model, project=args.project_name, log_model="all")

    # Dataset
    if args.time:
        dataset = temporal
        if args.model == "ginr":
            dataset = temporal_ginr
    else:
        dataset = spatial
        if args.model == "ginr":
            dataset = spatial_ginr

    # train - test - all dataset split
    train_dataset = dataset.Dataset(dataset_type="train", **vars(args))
    # valid_dataset = dataset.Dataset(dataset_type="valid", **vars(args))
    all_dataset = dataset.Dataset(dataset_type="all", **vars(args))
    
    # Calculate resolution of the dataset and make it argument
    args.lat_shape = all_dataset[0]["target_shape"][0]
    args.lon_shape = all_dataset[0]["target_shape"][1]
    resolution = abs(all_dataset[1]["lat"][2] - all_dataset[1]["lat"][1]) # latitude difference is resolution such as 0.25
    args.resolution =  round(float(resolution),4)
    
    if args.model == "ginr":
        args.input_dim = train_dataset.n_fourier + (1 if args.time else 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
     
    # val_loader = DataLoader(
    #     all_dataset,
    #     batch_size=len(all_dataset)//4,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    # )
    
    # Model
    model = model_dict[args.model].INR(all_dataset=all_dataset,**vars(args))

    # Pass scaler from dataset to model
    model.scaler = train_dataset.scaler
    model.normalize = train_dataset.zscore_normalize or train_dataset.normalize
    # model.all_dataset = all_dataset

    # Learning
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        monitor="full_train_psnr", mode="max", filename="best-{epoch}-{full_train_psnr:.2f}", save_top_k=1,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[lrmonitor_cb, checkpoint_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        # fast_dev_run=True,
    )
    
    trainer.fit(model, train_loader)
    # model.to('cpu')
    # val_loader = val_loader.to('cpu')
    # trainer.validate(model, val_loader)
        
    # elif args.task=='reg':
    #     trainer.fit(model, all_loader)
    #     trainer.test(model, all_loader, "best")
    
    
    best_model_path = checkpoint_cb.best_model_path
    print('best_model_path : ',best_model_path)
    if best_model_path:
        best_model = model.load_from_checkpoint(best_model_path, scaler=all_dataset.scaler, all_dataset=all_dataset, **vars(args))
        best_model.scaler = train_dataset.scaler
        best_model.normalize = train_dataset.zscore_normalize or train_dataset.normalize
        # model.all_dataset = all_dataset
    else:
        raise Exception("Best model is not saved properly")
    
    # trainer.test(model, all_loader)

    if args.plot:

        if "era5" in args.dataset_dir:
            # Currently only HR visualized
            parts = args.dataset_dir.split('/')
            if args.downscale_factor==2: # 0_50
                parts[1]='spatial_0_50'
                filename = '/'.join(parts) + '/data.nc'
            elif args.downscale_factor==4: # 1_00
                parts[1]='spatial_1_00'
                filename = '/'.join(parts) + '/data.nc'
            visualize_era5('train',
                train_dataset, best_model, filename, logger, args
            )
            
            visualize_era5('all',
                all_dataset, best_model, args.dataset_dir + "/data.nc", logger, args
            )
            # TODO : Visualize LR
            # Visualizing LR with earthmap is tricky due to interpolation
            # visualize_era5_LR(train_dataset, model, args.dataset_dir+"/data.nc", logger, args)
            # Maybe we don't need LR visualizations. At most, we would need the ground truth.
            # i.e., we don't need error map, prediction of LR visualizations
        elif '360' in args.dataset_dir:
            visualize_360('all',all_dataset, best_model, args, "HR", logger=logger)
            visualize_360('train',train_dataset, best_model, args, "LR", logger=logger)
        else:
            visualize_synthetic('all', all_dataset, best_model, args,"HR", logger=logger)
            visualize_synthetic('train', train_dataset, best_model, args,"LR", logger=logger)
            
