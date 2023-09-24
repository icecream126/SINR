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
from utils.utils import image_psnr

from scipy.interpolate import griddata
import numpy as np
from datetime import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

## Shared utilities ##

def draw_map(x, mode, variable, colormap, rmse, logger, args):
    # Plot flat map
    title = mode + "_" + variable
    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    x[variable].plot(ax=ax, cmap=colormap, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(title + ".png")
    logger.experiment.log(
        {title: wandb.Image(fig, caption=f"{args.model}:{rmse:.4f}(RMSE)")}
    )

    # Plot sphere map
    title = "sphere_" + mode + "_" + variable
    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot(
        111, projection=ccrs.Orthographic(central_longitude=20, central_latitude=40)
    )
    x[variable].plot(ax=ax, cmap=colormap, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(title + ".png")
    logger.experiment.log(
        {title: wandb.Image(fig, caption=f"{args.model}:{rmse:.4f}(RMSE)")}
    )
    if variable == "error":
        try:
            title = mode + "_" + variable + "_hist"
            fig = plt.figure(figsize=[12, 5])
            ax = fig.add_subplot(111)
            x[variable].plot.hist(ax=ax, bins=100)
            plt.savefig(title + ".png")
            logger.experiment.log(
                {title: wandb.Image(fig, caption=f"{args.model}:{rmse:.4f}(RMSE)")}
            )
        except:
            print("Failed to draw error histogram. Need Debugging.")


def draw_histogram(obj_type, filepath, logger):
    
    obj = np.load(filepath+'.npy')

    # Flatten the 2D error array to 1D
    obj = obj.flatten()

    # Round the error values to 4 decimal places
    # rounded_obj = np.round(flattened_obj, 4)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(obj, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title("Histogram of MSE Errors")
    ax.set_xlabel("Error")
    ax.set_ylabel("Number of Occurrences")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Log the plot directly to wandb without saving
    logger.experiment.log({obj_type+" Histogram": wandb.Image(fig)})

def array_to_image(arr, cmap='jet'):
    colormap = cm.get_cmap(cmap)
    normed_arr = (arr - arr.min()) / (arr.max() - arr.min())
    colored_arr = (colormap(normed_arr) * 255).astype(np.uint8)  # RGBA image
    return colored_arr

def get_fig(lon, lat, obj, cmap='jet'):
    plt.rcParams["font.size"] = 50
    fig = plt.figure(figsize=(40, 20))
    plt.tricontourf(
            lon,
            -lat,
            obj,
            levels=100,
            cmap=cmap,
        )
    plt.colorbar()
    return fig

## Visualization for each dataset ##

def visualize_synthetic(dtype, dataset, model, args, mode, logger):
    with torch.no_grad():
        data = dataset[:]

        inputs, target = data["inputs"], data["target"]

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]

        cart_inputs = to_cartesian(inputs)
        pred = model(cart_inputs)

        weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        error = torch.sum((pred - target) ** 2, dim=-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()

        lat = inputs[..., 0]
        lon = inputs[..., 1]
        deg_lat = torch.rad2deg(lat)  # set range [-90, 90]
        deg_lon = torch.rad2deg(lon)  # set range [-180, 180]
        lat, lon = lat.detach().cpu().numpy(), lon.detach().cpu().numpy()
        deg_lat, deg_lon = (
            deg_lat.detach().cpu().numpy(),
            deg_lon.detach().cpu().numpy(),
        )

        target_min, target_max = target.min(), target.max()
        target = (target - target_min) / (target_max - target_min)
        pred = (pred - target_min) / (target_max - target_min)
        pred = torch.clip(pred, 0, 1)

        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        error = error.squeeze(-1).detach().cpu().numpy()
        
        
        # Save target, pred, error as npy
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        
        filepath = 'output/'+str(current_datetime)+'_'+str(logger.experiment.id)+'/'
        os.makedirs(filepath, exist_ok=True)
        np.save(filepath+f'{dtype}_target', target)
        np.save(filepath+f'{dtype}_pred', pred)
        np.save(filepath+f'{dtype}_error', error)
        np.save(filepath+f'{dtype}_lat', lat)
        np.save(filepath+f'{dtype}_lon', lon)
        # draw_histogram('target', filepath+'target')
        # draw_histogram('pred', filepath+'pred')
        draw_histogram('error', filepath+f'{dtype}_error', logger)
        
        # target = (255 * target).astype(np.uint8)
        # pred = (255 * pred).astype(np.uint8)
        
        # target = array_to_image(target)
        # pred = array_to_image(pred)
        
        error = get_fig(lon, lat, error, cmap='hot')
        target = get_fig(lon, lat, target.flatten(), cmap='plasma')
        pred = get_fig(lon, lat, pred.flatten(), cmap='plasma')
        


        logger.experiment.log(
            {
                mode
                + " Error Map": wandb.Image(
                    error, caption=f"{args.model}: {rmse:.4f}(RMSE)"
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
        

def visualize_360(dtype, dataset, model, args, mode, logger):
    with torch.no_grad():
        data = dataset[:]

        inputs, target = data["inputs"], data["target"]

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]

        cart_inputs = to_cartesian(inputs)
        pred = model(cart_inputs)

        weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        error = torch.sum((pred - target) ** 2, dim=-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()

        lat = inputs[..., 0]
        lon = inputs[..., 1]
        deg_lat = torch.rad2deg(lat)  # set range [-90, 90]
        deg_lon = torch.rad2deg(lon)  # set range [-180, 180]
        lat, lon = lat.detach().cpu().numpy(), lon.detach().cpu().numpy()
        deg_lat, deg_lon = (
            deg_lat.detach().cpu().numpy(),
            deg_lon.detach().cpu().numpy(),
        )

        target_min, target_max = target.min(), target.max()
        target = (target - target_min) / (target_max - target_min)
        pred = (pred - target_min) / (target_max - target_min)
        pred = torch.clip(pred, 0, 1)

        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        error = error.squeeze(-1).detach().cpu().numpy()
        
        # Save target, pred, error as npy
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        
        filepath = 'output/'+str(current_datetime)+'_'+str(logger.experiment.id)+'/'
        os.makedirs(filepath, exist_ok=True)
        np.save(filepath+f'{dtype}_target', target)
        np.save(filepath+f'{dtype}_pred', pred)
        np.save(filepath+f'{dtype}_error', error)
        np.save(filepath+f'{dtype}_lat', lat)
        np.save(filepath+f'{dtype}_lon', lon)
        
        # draw_histogram('target', filepath+'target')
        # draw_histogram('pred', filepath+'pred')
        draw_histogram('error', filepath+f'{dtype}_error', logger)
        
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

    

def visualize_era5(dtype, dataset, model, filename, logger, args):
    with torch.no_grad():
        data = dataset[:]

        inputs, target = data["inputs"], data["target"]

        mean_lat_weight = data["mean_lat_weight"]
        target_shape = data["target_shape"]
        if args.model != "ginr":
            proceed_inputs = to_cartesian(inputs)
            lat = inputs[..., :1]
        else:
            lat = data["spherical"][..., :1]
            proceed_inputs = inputs

        pred = model(proceed_inputs)

        weights = torch.abs(torch.cos(lat))
        weights = weights / mean_lat_weight
        if weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        error = torch.sum((pred - target) ** 2, dim=-1)
        error = weights * error
        loss = error.mean()
        rmse = torch.sqrt(loss).item()

        target_min, target_max = target.min(), target.max()
        target = (target - target_min) / (target_max - target_min)
        pred = (pred - target_min) / (target_max - target_min)
        pred = torch.clip(pred, 0, 1)

        target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
        error = error.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()

        # Save target, pred, error as npy
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        
        filepath = 'output/'+str(current_datetime)+'_'+str(logger.experiment.id)+'/'
        os.makedirs(filepath, exist_ok=True)
        
        np.save(filepath+f'{dtype}_target', target)
        np.save(filepath+f'{dtype}_pred', pred)
        np.save(filepath+f'{dtype}_error', error)
        np.save(filepath+f'{dtype}_lat', lat)
        np.save(filepath+f'{dtype}_lon', lon)
        
        # draw_histogram('target', filepath+'target')
        # draw_histogram('pred', filepath+'pred')
        draw_histogram('error', filepath+f'{dtype}_error', logger)
        
        
        dims = ("latitude", "longitude")

        x = xr.open_dataset(filename)

        # Assign prediction and error for visualization
        x = x.assign(pred=(dims, pred))
        x = x.assign(error=(dims, error))

        if "geopotential" in args.dataset_dir:
            ground_truth = "z"
            colormap = "YlOrBr_r"
        elif "wind" in args.dataset_dir:
            ground_truth = "u"
            colormap = "Greys_r"
        elif "cloud" in args.dataset_dir:
            ground_truth = "cc"
            colormap = "PuBu_r"

        # FIXME: Implement saving raw data

        # Assign normalized target variable for visualization
        gt_min, gt_max = float(x[ground_truth].min()), float(x[ground_truth].max())
        x = x.assign(target=(x[ground_truth] - gt_min) / (gt_max - gt_min))

        draw_map(x, "HR", "pred", colormap, rmse, logger, args)
        draw_map(x, "HR", "error", "hot", rmse, logger, args)
        draw_map(x, "HR", "target", colormap, rmse, logger, args)
