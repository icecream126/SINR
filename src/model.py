import wandb
import torch
from math import pi
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler

from utils.psnr import mse2psnr
import matplotlib.pyplot as plt
from utils.change_coord_sys import to_spherical
from inrs import relu, siren, wire, shinr, swinr

model_dict = {
    'relu': relu,
    'siren': siren,
    'wire': wire,
    'shinr': shinr,
    'swinr': swinr
}

class INR(pl.LightningModule):
    def __init__(
            self,
            dataset: str,
            model: str,
            validation: bool,
            lr: float=0.001,
            lr_patience: int=500,
            plot: bool=False,
            **kwargs
        ):
        super().__init__()

        self.dataset = dataset
        self.name = model
        self.model = model_dict[model].INR(**kwargs)
        self.plot = plot
        self.validation = validation
        self.lr = lr
        self.lr_patience = lr_patience

        self.sync_dist = torch.cuda.device_count() > 1

        self.loss_fn = nn.MSELoss()
        self.min_valid_loss = None

    def forward(self, points):
        return self.model(points)
    
    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train_psnr", mse2psnr(loss), sync_dist=self.sync_dist)
        return loss
    
    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("valid_psnr", mse2psnr(loss), sync_dist=self.sync_dist)
        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        
        self.log("test_mse", loss)
        self.log("test_psnr", mse2psnr(loss))

        if self.validation:
            self.log("min_valid_loss", self.min_valid_loss)
        
        if self.plot:
            self.visualize(inputs, target, pred, loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler, 
            "monitor": 'valid_loss' if self.validation else 'train_loss', 
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}

    def visualize(self, inputs, target, pred, loss):
        dist = nn.PairwiseDistance(eps=0)

        if self.name != 'shinr':
            inputs = to_spherical(inputs)

        inputs = inputs[0]
        target = target[0]
        pred = pred[0]

        lat = inputs[..., 0].detach().cpu().numpy()
        lon = inputs[..., 1].detach().cpu().numpy()
        error = dist(target, pred).detach().cpu().numpy()

        fig = plt.figure(figsize=(40, 20))

        plt.tricontourf(
            lon,
            pi - lat,
            error,
            cmap = 'hot',
        )

        plt.title(f'{self.name} Error map(PSNR: {mse2psnr(loss):.2f})', fontsize=40)
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()
        plt.savefig(f'./figure/{self.dataset}_{self.name}.png')

        wandb.log({"Error Map": wandb.Image(fig)})

        if self.dataset in ['era5', 'dpt2m', 'gustsfc', 'tcdcclm']:
            plt.clf()
            target = target.squeeze(-1).detach().cpu().numpy()
            fig = plt.figure(figsize=(40, 20))

            plt.tricontourf(
                lon,
                pi - lat,
                target,
                cmap = 'hot',
            )

            plt.title(f'{self.name} Ground Truth', fontsize=40)
            plt.colorbar()
            plt.show()
            plt.savefig(f'./figure/{self.dataset}_{self.name}_gt.png')

            wandb.log({"Ground Truth": wandb.Image(fig)})