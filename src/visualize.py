import wandb
import torch
import matplotlib.pyplot as plt

from utils.utils import mse2psnr


def error_map(dataset, model, args):
    data = dataset[:]
    inputs, target = data["inputs"], data["target"]

    weights = torch.cos(inputs[..., :1])
    weights = weights / weights.mean()

    pred = model(inputs)

    error = torch.sum((pred-target)**2, dim=-1, keepdim=True)
    error = weights * error
    loss = error.mean()

    lat = inputs[..., 0].detach().cpu().numpy()
    lon = inputs[..., 1].detach().cpu().numpy()
    error = torch.log10(error+1e-6).squeeze(-1).detach().cpu().numpy()

    if 'spatial' in args.dataset_path:
        fig = plt.figure(figsize=(40, 20))
        plt.rcParams.update({'font.size': 50})

        plt.tricontourf(
            lon,
            -lat,
            error,
            cmap = 'hot',
        )

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=50)

        plt.title(f'{args.model} (PSNR: {mse2psnr(loss):.2f})', fontsize=60)
        plt.clim(-5, -1)
        plt.show()

        wandb.log({"Error Map": wandb.Image(fig)})

        if 'sun360' in args.dataset_path:
            wandb.log({"Ground Truth": wandb.Image(args.dataset_path)})

        else:
            plt.clf()
            target = target.squeeze(-1).detach().cpu().numpy()
            fig = plt.figure(figsize=(40, 20))
            plt.rcParams.update({'font.size': 50})

            plt.tricontourf(
                lon,
                -lat,
                target,
                cmap = 'hot',
            )

            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=50)

            plt.title(f'Ground Truth', fontsize=60)
            plt.clim(0, 1)
            plt.show()

            wandb.log({"Ground Truth": wandb.Image(fig)})
    
    elif 'temporal' in args.dataset_path:
        pass # TODO: make error map video