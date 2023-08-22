import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize(dataset, model, args, mode):
    data = dataset[:]

    inputs, target = data["inputs"], data["target"]
    mean_lat_weight = data['mean_lat_weight']
    target_shape = data['target_shape']

    pred = model(inputs)

    weights = torch.cos(inputs[..., :1])
    weights = weights / mean_lat_weight

    error = torch.sum((pred-target)**2, dim=-1, keepdim=True)
    error = weights * error
    loss = error.mean()
    rmse = torch.sqrt(loss).item()

    target_min, target_max = target.min(), target.max()
    target = (target - target_min) / (target_max - target_min)
    pred = (pred - target_min) / (target_max - target_min)
    pred = torch.clip(pred, 0, 1)

    lat = inputs[..., 0].detach().cpu().numpy()
    lon = inputs[..., 1].detach().cpu().numpy()
    target = target.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
    pred = pred.reshape(*target_shape).squeeze(-1).detach().cpu().numpy()
    error = error.squeeze(-1).detach().cpu().numpy() 

    target = (255*target).astype(np.uint8)
    pred = (255*pred).astype(np.uint8)

    plt.rcParams['font.size'] = 50
    fig = plt.figure(figsize=(40, 20))
    plt.tricontourf(
        lon,
        -lat,
        error,
        levels=100,
        cmap='hot',
    )
    plt.colorbar()

    wandb.log({
        mode + " Error Map": wandb.Image(fig, caption=f'{args.model}: {rmse:.4f}(RMSE)'),
        mode + " Prediction": wandb.Image(pred, caption=f'{args.model}: {rmse:.4f}(RMSE)'),
        mode + " Truth": wandb.Image(target, caption=f'{args.model}: {rmse:.4f}(RMSE)'),
    })