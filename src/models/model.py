from typing import List, Union

# from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import pytorch_lightning as pl
from utils.utils import to_cartesian, mse2psnr
from torch.optim import lr_scheduler


class MODEL(pl.LightningModule):
    def __init__(self, lr, lr_patience, **kwargs):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience
        self.scaler = None
        self.target_normalize = False

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]  # [512, 3], [512, 3]
        mean_lat_weight = data["mean_lat_weight"]  # [512] with 0.6341

        weights = torch.abs(torch.cos(inputs[..., :1]))  # [512, 1]
        weights = weights / mean_lat_weight  # [512, 512]

        if self.time:
            inputs = torch.cat((to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1)
        else:
            inputs = to_cartesian(inputs)

        pred = self.forward(inputs)  # [512, 3]

        error = torch.sum(
            (pred - target) ** 2, dim=-1, keepdim=True
        )  # [512, 1] with 0.9419
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error  # [512, 512] with 1.2535

        loss = error.mean()  # 1.2346

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(pred)
            target = self.scaler.inverse_transform(target)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            if len(error.shape) > len(weights.shape):
                error = error.squeeze(-1)
            mse = weights * error
            mse = mse.mean()
        else:
            mse = loss

        rmse = torch.sqrt(mse)
        w_psnr_val = mse2psnr(mse.detach().cpu().numpy())

        self.log("batch_train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_train_mse", mse, prog_bar=True, sync_dist=True)
        self.log("batch_train_psnr", w_psnr_val, prog_bar=True, sync_dist=True)
        self.log("batch_train_rmse", rmse, prog_bar=True, sync_dist=True)

        return {"loss": loss, "batch_train_psnr": w_psnr_val.item()}

    def train_epoch_end(self, outputs):
        avg_train_mse = torch.stack([torch.tensor(x["loss"]) for x in outputs]).mean()
        avg_train_psnr = torch.stack(
            [torch.tensor(x["batch_train_psnr"]) for x in outputs]
        ).mean()
        self.log("avg_train_mse", avg_train_mse)
        self.log("avg_train_psnr", avg_train_psnr, prog_bar=True, sync_dist=True)

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]

        weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight

        if self.time:
            inputs = torch.cat((to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1)
        else:
            inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        loss = error.mean()

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(pred)
            target = self.scaler.inverse_transform(target)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            if len(error.shape) > len(weights.shape):
                error = error.squeeze(-1)
            mse = weights * error
            mse = mse.mean()
        else:
            mse = loss

        rmse = torch.sqrt(mse)
        w_psnr_val = mse2psnr(mse.detach().cpu().numpy())

        self.log("batch_test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_test_mse", mse, prog_bar=True, sync_dist=True)
        self.log("batch_test_rmse", rmse)
        self.log("batch_test_mse", loss, prog_bar=True, sync_dist=True)
        self.log("batch_test_psnr", w_psnr_val)
        return {
            "batch_test_mse": loss,
            "batch_test_psnr": w_psnr_val.item(),
            "batch_test_rmse": rmse.item(),
        }

    def test_epoch_end(self, outputs):
        # Compute the average of test_mse and batch_test_psnr over the entire epoch
        avg_test_mse = torch.stack(
            [torch.tensor(x["batch_test_mse"]) for x in outputs]
        ).mean()
        avg_test_psnr = torch.stack(
            [torch.tensor(x["batch_test_psnr"]) for x in outputs]
        ).mean()
        avg_test_rmse = torch.stack(
            [torch.tensor(x["batch_test_rmse"]) for x in outputs]
        ).mean()

        # Log the computed averages
        self.log("avg_test_rmse", avg_test_rmse, prog_bar=True, sync_dist=True)
        self.log("avg_test_mse", avg_test_mse, prog_bar=True, sync_dist=True)
        self.log("avg_test_psnr", avg_test_psnr, prog_bar=True, sync_dist=True)
        self.log(
            "final_test_psnr",
            mse2psnr(avg_test_mse.detach().cpu().numpy()),
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler,
            "monitor": "batch_train_mse",
            "frequency": 1,
        }
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

        weights = torch.abs(
            torch.cos(inputs[..., 0])
        )  # [512, 1] # make sure nonzero weights
        weights = weights / mean_lat_weight  # [512, 512]

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1)  # , keepdim=True)
        error_orig = torch.sum((pred - g_target) ** 2, dim=-1)  # , keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        error_orig = weights * error_orig
        loss = error.mean()
        loss_orig = error_orig.mean()

        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())
        g_w_psnr_val = mse2psnr(loss_orig.detach().cpu().numpy())

        # self.log("batch_train_mse", loss, prog_bar=True)
        # self.log("batch_train_mse_orig", loss_orig, prog_bar=True)
        self.log("batch_train_psnr", w_psnr_val)
        self.log("batch_train_psnr_orig", g_w_psnr_val)

        return {"loss": loss, "batch_train_mse_orig": loss_orig}

    def training_epoch_end(self, outputs):
        # avg_train_mse = torch.stack(
        #     [torch.tensor(x["loss"]) for x in outputs]
        # ).mean()
        avg_train_mse_orig = torch.stack(
            [torch.tensor(x["batch_train_mse_orig"]) for x in outputs]
        ).mean()
        # self.log("avg_train_mse", avg_train_mse)
        self.log("avg_train_mse_orig", avg_train_mse_orig)
        self.log(
            "final_train_psnr_orig",
            mse2psnr(avg_train_mse_orig.detach().cpu().numpy()),
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, data, batch_idx):
        inputs, target, g_target = data["inputs"], data["target"], data["g_target"]
        mean_lat_weight = data["mean_lat_weight"]

        weights = torch.abs(torch.cos(inputs[..., 0]))  # make sure nonnegative weights
        weights = weights / mean_lat_weight

        inputs = to_cartesian(inputs)
        pred = self.forward(inputs)

        error = torch.sum((pred - target) ** 2, dim=-1)  # , keepdim=True)
        error_orig = torch.sum((pred - g_target) ** 2, dim=-1)  # , keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        error_orig = weights * error_orig
        loss = error.mean()
        loss_orig = error_orig.mean()

        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())
        g_w_psnr_val = mse2psnr(loss_orig.detach().cpu().numpy())

        # self.log("batch_test_mse", loss)
        # self.log("batch_test_mse_orig", loss_orig)
        self.log("batch_test_psnr", w_psnr_val)
        self.log("batch_test_psnr_orig", g_w_psnr_val)
        return {
            "batch_test_mse": loss.item(),
            "batch_test_mse_orig": loss_orig.item(),
            "batch_test_psnr": w_psnr_val.item(),
            "batch_test_psnr_orig": g_w_psnr_val.item(),
        }

    def test_epoch_end(self, outputs):
        # Compute the average of test_mse and batch_test_psnr over the entire epoch
        avg_test_mse = torch.stack(
            [torch.tensor(x["batch_test_mse"]) for x in outputs]
        ).mean()
        avg_test_mse_orig = torch.stack(
            [torch.tensor(x["batch_test_mse_orig"]) for x in outputs]
        ).mean()
        avg_batch_test_psnr = torch.stack(
            [torch.tensor(x["batch_test_psnr"]) for x in outputs]
        ).mean()
        avg_batch_test_psnr_orig = torch.stack(
            [torch.tensor(x["batch_test_psnr_orig"]) for x in outputs]
        ).mean()

        # Log the computed averages
        self.log("avg_test_mse", avg_test_mse, prog_bar=True, sync_dist=True)
        self.log("avg_test_mse_orig", avg_test_mse_orig, prog_bar=True, sync_dist=True)
        self.log("avg_test_psnr", avg_batch_test_psnr, prog_bar=True, sync_dist=True)
        self.log(
            "avg_test_psnr_orig",
            avg_batch_test_psnr_orig,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "final_test_psnr",
            mse2psnr(avg_test_mse.detach().cpu().numpy()),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "final_test_psnr_orig",
            mse2psnr(avg_test_mse_orig.detach().cpu().numpy()),
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler,
            "monitor": "avg_train_mse_orig",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}
