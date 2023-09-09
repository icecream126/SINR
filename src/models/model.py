import torch
import pytorch_lightning as pl
from utils.utils import to_cartesian, psnr
from torch.optim import lr_scheduler


class MODEL(pl.LightningModule):
    def __init__(self, lr, lr_patience, **kwargs):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]  # [512, 3], [512, 3]
        mean_lat_weight = data["mean_lat_weight"]  # [512] with 0.6341

        weights = torch.cos(inputs[..., 0])  # [512, 1]
        weights = weights / mean_lat_weight  # [512, 512]

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)  # [512, 3]

        error = torch.sum(
            (pred - target) ** 2, dim=-1, keepdim=True
        )  # [512, 1] with 0.9419
        error = weights * error  # [512, 512] with 1.2535
        loss = error.mean()  # 1.2346
        
        w_psnr_val = psnr(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "batch_train_psnr",
            w_psnr_val,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]

        weights = torch.cos(inputs[..., 0])
        weights = weights / mean_lat_weight

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error = weights * error
        loss = error.mean()

        w_psnr_val = psnr(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_valid_psnr", w_psnr_val)
        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]

        weights = torch.cos(inputs[..., 0])
        weights = weights / mean_lat_weight

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error = weights * error
        loss = error.mean()

        w_psnr_val = psnr(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        
        self.log("test_mse", loss, prog_bar=True, sync_dist=True)
        self.log("batch_test_psnr", w_psnr_val)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {"scheduler": scheduler, "monitor": "valid_loss", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}


class DENOISING_MODEL(pl.LightningModule):
    def __init__(self, lr, lr_patience, model, **kwargs):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience
        self.model = model

    def training_step(self, data, batch_idx):
        inputs, target, g_target = (
            data["inputs"],
            data["target"],
            data["g_target"],
        )  # [512, 3] for each
        mean_lat_weight = data["mean_lat_weight"]  # 512

        weights = torch.cos(inputs[..., 0])  # [512, 1]
        weights = weights / mean_lat_weight  # [512, 512]

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error_orig = torch.sum((pred - g_target) ** 2, dim=-1, keepdim=True)
        error = weights * error
        error_orig = weights * error_orig
        loss = error.mean()
        loss_orig = error_orig.mean()
        
        w_psnr_val = psnr(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        g_w_psnr_val = psnr(pred.detach().cpu().numpy(), g_target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_orig", loss_orig, prog_bar=True)
        self.log("batch_train_w_psnr", w_psnr_val)
        self.log("batch_train_w_psnr_orig", g_w_psnr_val)

        return loss

    def test_step(self, data, batch_idx):
        inputs, target, g_target = data["inputs"], data["target"], data["g_target"]
        mean_lat_weight = data["mean_lat_weight"]

        weights = torch.cos(inputs[..., 0])
        weights = weights / mean_lat_weight

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error_orig = torch.sum((pred - g_target) ** 2, dim=-1, keepdim=True)
        error = weights * error
        error_orig = weights * error_orig
        loss = error.mean()
        loss_orig = error_orig.mean()

        w_psnr_val = psnr(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        g_w_psnr_val = psnr(pred.detach().cpu().numpy(), g_target.detach().cpu().numpy(), weights.unsqueeze(-1).detach().cpu().numpy())
        
        self.log("test_mse", loss)
        self.log("test_mse_orig", loss_orig)
        self.log("batch_test_w_psnr", w_psnr_val)
        self.log("batch_test_w_psnr_orig", g_w_psnr_val)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler,
            "monitor": "train_loss_orig",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}
