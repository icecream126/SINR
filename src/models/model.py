import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler

class MODEL(pl.LightningModule):
    def __init__(
            self,
            lr,
            lr_patience,
            **kwargs
        ):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience
    
    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data['mean_lat_weight']

        weights = torch.cos(inputs[..., :1])
        weights = weights / mean_lat_weight

        pred = self.forward(inputs)

        error = torch.sum((pred-target)**2, dim=-1, keepdim=True)
        error = weights * error
        loss = error.mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data['mean_lat_weight']

        weights = torch.cos(inputs[..., :1])
        weights = weights / mean_lat_weight

        pred = self.forward(inputs)

        error = torch.sum((pred-target)**2, dim=-1, keepdim=True)
        error = weights * error
        loss = error.mean()

        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data['mean_lat_weight']

        weights = torch.cos(inputs[..., :1])
        weights = weights / mean_lat_weight

        pred = self.forward(inputs)

        error = torch.sum((pred-target)**2, dim=-1, keepdim=True)
        error = weights * error
        loss = error.mean()

        self.log("test_mse", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {"scheduler": scheduler, "monitor": 'valid_loss', "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}