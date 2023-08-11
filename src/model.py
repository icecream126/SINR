import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler

from inrs import relu, siren, wire, shinr, swinr
from utils.psnr import mse2psnr

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
            model: str,
            lr: float=0.001,
            lr_patience: int=500,
            **kwargs
        ):
        super().__init__()
        self.lr = lr
        self.lr_patience = lr_patience
        self.model = model_dict[model].INR(**kwargs)
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
        self.log("min_valid_loss", self.min_valid_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {"scheduler": scheduler, "monitor": 'valid_loss', "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}