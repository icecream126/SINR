import torch
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
            plot: bool,
            lr: float=0.001,
            lr_patience: int=500,
            **kwargs
        ):
        super().__init__()

        self.dataset = dataset
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
            self.visualize(inputs, target, pred)
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

    def visualize(self, inputs, target, pred):
        dist = nn.PairwiseDistance(eps=0)

        if self.model != 'shinr':
            inputs = to_spherical(inputs)
        if self.dataset not in ['sun360', 'circle']:
            inputs = inputs[0]
            target = target[0]
            pred = pred[0]

        lat = inputs[..., 0].detach().cpu().numpy()
        lon = inputs[..., 1].detach().cpu().numpy()
        error = dist(target, pred).detach().cpu().numpy()

        # 그래프 생성
        plt.tricontourf(
            x = lat,
            y = lon,
            Z = error,
            cmap = 'hot',
        )

        plt.show()
        plt.savefig(f'./figure/{self.dataset}/error.png')