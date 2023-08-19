import wandb

from math import pi
from torch import nn
import matplotlib.pyplot as plt

from utils.psnr import mse2psnr
from utils.change_coord_sys import to_spherical


def error_map(dataset, model, args):
    dist = nn.PairwiseDistance(eps=0)
    data = dataset[:]
    inputs, target = data["inputs"], data["target"]

    if args.model != 'shinr':
        inputs = to_spherical(inputs)

    pred = model(inputs)
    loss = model.loss_fn(pred, target)

    lat = inputs[..., 0].detach().cpu().numpy()
    lon = inputs[..., 1].detach().cpu().numpy()
    error = dist(target, pred).detach().cpu().numpy()

    if 'spatial' in args.dataset_path:
        fig = plt.figure(figsize=(40, 20))

        plt.tricontourf(
            lon,
            pi - lat,
            error,
            cmap = 'hot',
        )

        plt.title(f'{args.model} (PSNR: {mse2psnr(loss.item()):.2f})', fontsize=60)
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()

        wandb.log({"Error Map": wandb.Image(fig)})

        if 'sun360' in args.dataset_path:
            wandb.log({"Ground Truth": wandb.Image(args.dataset_path)})

        else:
            plt.clf()
            target = target.squeeze(-1).detach().cpu().numpy()
            fig = plt.figure(figsize=(40, 20))

            plt.tricontourf(
                lon,
                pi - lat,
                target,
                cmap = 'hot',
            )

            plt.title(f'{args.model} Ground Truth', fontsize=60)
            plt.colorbar()
            plt.show()

            wandb.log({"Ground Truth": wandb.Image(fig)})
    
    elif 'temporal' in args.dataset_path:
        pass # TODO: make error map video