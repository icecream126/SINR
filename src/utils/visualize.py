import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from utils.utils import to_cartesian
from PIL import Image as PILImage
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.cm as cm
import math

from scipy.interpolate import griddata
import numpy as np


def visualize_360(dataset, model, args, mode, logger):
    with torch.no_grad():
        data = dataset[:]

        inputs, target = data["inputs"], data["target"]

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]

        cart_inputs = to_cartesian(inputs)
        pred = model(cart_inputs)

        weights = torch.cos(inputs[..., :1])
        weights = weights / mean_lat_weight
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        error = torch.sum((pred - target) ** 2, dim=-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()


        lat = inputs[..., 0]
        lon = inputs[..., 1]
        deg_lat = torch.rad2deg(lat) # set range [-90, 90]
        deg_lon = torch.rad2deg(lon) # set range [-180, 180]
        lat, lon = lat.detach().cpu().numpy(), lon.detach().cpu().numpy()
        deg_lat, deg_lon = deg_lat.detach().cpu().numpy(), deg_lon.detach().cpu().numpy()
        
        target_min, target_max = target.min(), target.max()
        pred_min, pred_max = pred.min(), pred.max()
        target = (target - target_min) / (target_max - target_min)
        pred = (pred - pred_min) / (pred_max - pred_min)
        pred = torch.clip(pred, 0, 1)
        
        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        error = error.squeeze(-1).detach().cpu().numpy()
            
        target = (255 * target).astype(np.uint8)
        pred = (255 * pred).astype(np.uint8)  
        plt.rcParams["font.size"] = 50
        fig = plt.figure(figsize=(40, 20))
        plt.tricontourf(
            lon,
            -lat,
            error,
            levels=100,
            cmap="hot",
        )
        plt.colorbar()

        logger.experiment.log(
            {
                mode
                + " Error Map": wandb.Image(
                    fig, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                ),
                mode
                + " Prediction": wandb.Image(
                    pred, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                ),
                mode
                + " Truth": wandb.Image(
                    target, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                ),
            }
        )

def draw_map(x, mode, variable, colormap, rmse, logger, args):
    # Plot flat map
    title = mode+'_'+variable
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    x[variable].plot(ax=ax,cmap=colormap, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(title+'.png')
    logger.experiment.log({
        title: wandb.Image(fig, caption=f"{args.model}:{rmse:.4f}(RMSE)")
    })
    
    # Plot sphere map
    title = 'sphere_'+mode+'_'+variable
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_longitude=20, central_latitude=40))
    x[variable].plot(ax=ax,cmap=colormap, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(title+'.png')
    logger.experiment.log({
        title: wandb.Image(fig, caption=f"{args.model}:{rmse:.4f}(RMSE)")
    })



def visualize_era5(dataset, model, filename, logger, args):
    with torch.no_grad():
        data = dataset[:]

        inputs, target = data["inputs"], data["target"]

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]

        cart_inputs = to_cartesian(inputs)
        pred = model(cart_inputs)

        weights = torch.cos(inputs[..., :1])
        weights = weights / mean_lat_weight
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        error = torch.sum((pred - target) ** 2, dim=-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()
        
        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        error = error.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        
        dims=('latitude','longitude')


        x = xr.open_dataset(filename)
        
        # Assign prediction and error for visualization        
        x = x.assign(pred=(dims,pred))
        x = x.assign(error=(dims,error))
        
        if "geopotential" in args.dataset_dir:
            ground_truth='z'
            colormap='YlOrBr_r'
        elif "wind" in args.dataset_dir:
            ground_truth='u'
            colormap='Greys_r'
        elif "cloud" in args.dataset_dir:
            ground_truth='cc'
            colormap='PuBu_r'
        
        # Assign normalized target variable for visualization
        gt_min, gt_max = float(x[ground_truth].min()),float(x[ground_truth].max())
        x = x.assign(target=(x[ground_truth]-gt_min)/(gt_max-gt_min))
            
        draw_map(x, "HR", "pred" , colormap, rmse, logger, args)
        draw_map(x, "HR", "error" , "hot", rmse, logger, args)
        draw_map(x, "HR", "target" , colormap, rmse, logger, args)
        
        


def visualize_denoising(dataset, model, args, mode="denoising", logger=None):
    with torch.no_grad():
        data = dataset.get_all_data()

        inputs, target, g_target = data["inputs"].unsqueeze(0), data["target"].unsqueeze(0), data["g_target"].unsqueeze(0)

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]

        weights = torch.cos(inputs[..., 0])
        weights = weights / mean_lat_weight

        cart_inputs = to_cartesian(inputs)
        pred = model(cart_inputs)

        # Noisy - Prediction Error
        error = torch.sum((pred - target) ** 2, dim=-1)
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()

        # Ground Truth - Prediction Error
        g_error = torch.sum((pred - g_target) ** 2, dim=-1)
        g_error = weights * g_error
        g_loss = g_error.mean()
        g_rmse = torch.sqrt(g_loss).item()

        # Noisy normalization
        target_min, target_max = target.min(), target.max()
        target = (target - target_min) / (target_max - target_min)
        pred = (pred - target_min) / (target_max - target_min)
        pred = torch.clip(pred, 0, 1)

        # Ground Truth normalization
        g_target_min, g_target_max = g_target.min(), g_target.max()
        g_target = (g_target - g_target_min) / (g_target_max - g_target_min)

        lat = inputs[..., 0].squeeze(0).detach().cpu().numpy()
        lon = inputs[..., 1].squeeze(0).detach().cpu().numpy()
        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        if error.shape[-1]==1:
            error = error.squeeze(-1)
        if error.shape[0]==1:
            error = error.squeeze(0)
            
        error = error.detach().cpu().numpy()

        g_target = g_target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        if g_error.shape[-1]==1:
            g_error = g_error.squeeze(-1)
        if g_error.shape[0]==1:
            g_error = g_error.squeeze(0)
            
        g_error = g_error.detach().cpu().numpy()
        

        # multiply 255
        target = (255 * target).astype(np.uint8)
        pred = (255 * pred).astype(np.uint8)
        g_target = (255 * g_target).astype(np.uint8)

        pil_target = PILImage.fromarray(target)
        pil_g_target = PILImage.fromarray(g_target)
        pil_pred = PILImage.fromarray(pred)

        # pil_target.save("pil_noisy_img.png")
        # pil_g_target.save("pil_gt_img.png")
        # pil_pred.save("pil_pred.png")

        plt.rcParams["font.size"] = 50
        fig1 = plt.figure(figsize=(40, 20))
        plt.tricontourf(
            lon,
            -lat,
            error,
            levels=100,
            cmap="hot",
        )
        plt.colorbar()
        logger.experiment.log(
            {
                mode
                + " Error Map": wandb.Image(
                    fig1, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                )
            }
        )

        plt.rcParams["font.size"] = 50
        fig2 = plt.figure(figsize=(40, 20))
        plt.tricontourf(
            lon,
            -lat,
            g_error,
            levels=100,
            cmap="hot",
        )
        plt.colorbar()

        logger.experiment.log(
            {
                mode
                + " GT Error Map": wandb.Image(
                    fig2, caption=f"{args.model}: {rmse:.4f}(G_RMSE)"
                ),
                mode
                + " Prediction": wandb.Image(
                    pil_pred, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                ),
                mode
                + " Noisy Truth": wandb.Image(
                    pil_target, caption=f"{args.model}: {rmse:.4f}(RMSE)"
                ),
                mode
                + " Ground Truth": wandb.Image(
                    pil_g_target, caption=f"{args.model}: {g_rmse:.4f}(G_RMSE)"
                ),
            }
        )
