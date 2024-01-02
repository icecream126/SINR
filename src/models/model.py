from typing import List, Union

# from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import pytorch_lightning as pl
from utils.utils import to_cartesian, mse2psnr, _ws_ssim
from torch.optim import lr_scheduler

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

class MODEL(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        lr,
        lr_patience,
        model,
        normalize,
        zscore_normalize,
        all_dataset,
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience
        self.normalize = normalize or zscore_normalize
        self.scaler = None
        self.target_normalize = False
        self.model = model
        self._device = None
        self.all_dataset = all_dataset
        self.last_full_train_psnr = None
        self.last_full_train_rmse = None
        self.best_full_train_psnr = 0
        self.input_dim = input_dim

    def metric_all(self, device, mode="train"):
        with torch.no_grad():
            all_data = self.all_dataset[:]

            all_inputs, all_target = all_data["inputs"].to(device), all_data[
                "target"
            ].to(device)
            mean_lat_weight = all_data["mean_lat_weight"].to(device)

            # if self.model != "ginr" and self.model!='learnable' and self.model!='coolchic_interp' and self.model!='ngp_interp':
            if self.input_dim == 3:
                proceed_inputs = to_cartesian(all_inputs)
                lat = all_inputs[..., :1]
            else:
                lat = all_inputs[..., :1]
                proceed_inputs = all_inputs

            all_pred = self(proceed_inputs)  # self.forward?

            weights = torch.abs(torch.cos(lat))
            weights = weights / mean_lat_weight
            if weights.shape[-1] == 1:
                weights = weights.squeeze(-1)

            error = torch.sum((all_pred - all_target) ** 2, dim=-1)
            error = weights * error
            all_loss = error.mean()
            all_rmse = torch.sqrt(all_loss).item()
            all_w_psnr_val = mse2psnr(all_loss.detach().cpu().numpy())

            # self.last_full_train_psnr = all_w_psnr_val
            # self.last_full_train_rmse = all_rmse
            if mode == "train" and self.best_full_train_psnr < all_w_psnr_val:
                self.best_full_train_psnr = all_w_psnr_val

            self.log(
                "full_" + mode + "_rmse",
                all_rmse,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "full_" + mode + "_psnr",
                all_w_psnr_val,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]  # [512, 3], [512, 3]
        mean_lat_weight = data["mean_lat_weight"]  # [512] with 0.6341

        # Weights 먼저 구해주고
        if self.input_dim == 2:
            # if self.model=='learnable' or self.model=='coolchic_interp' or self.model=='ngp_interp':
            rad = torch.deg2rad(inputs)
            rad_lat = rad[..., :1]

            weights = torch.abs(torch.cos(rad_lat))
        else:
            weights = torch.abs(torch.cos(inputs[..., :1]))  # [512, 1]
        weights = weights / mean_lat_weight  # [512, 512]

        # Model에 들어갈 input 계산
        if self.time:
            if self.input_dim == 3:
                # if self.model!='learnable' and self.model!='coolchic_interp' and self.model!='ngp_interp':
                inputs = torch.cat(
                    (to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1
                )

        else:
            if self.input_dim == 3:
                # if self.model != 'swinr_pe' and self.model!='learnable' and self.model!='coolchic_interp' and self.model!='ngp_interp':
                inputs = to_cartesian(inputs)

        pred = self.forward(inputs)  # [512, 3]
        self._device = pred.device

        error = torch.sum(
            (pred - target) ** 2, dim=-1, keepdim=True
        )  # [512, 1] with 0.9419
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error  # [512, 512] with 1.2535

        loss = error.mean()  # 1.2346 # normalize 된 data에서의 loss

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(
                pred
            )  # scale prediction to raw data scale (unnormalize)
            target = self.scaler.inverse_transform(
                target
            )  # scale target     to raw data scale (unnormalize)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            if len(error.shape) > len(weights.shape):
                error = error.squeeze(-1)
            mse = weights * error
            mse = mse.mean()
            rmse = torch.sqrt(mse)
            psnr = mse2psnr(mse.detach().cpu().numpy())

            self.log("train_unnorm_rmse", rmse, prog_bar=True, sync_dist=True)
            self.log("train_unnorm_mse", mse, prog_bar=True, sync_dist=True)
            self.log("train_unnorm_psnr", psnr, prog_bar=True, sync_dist=True)

        rmse = torch.sqrt(loss)
        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())
        # w_ssim_val = _ws_ssim(pred, target)

        self.log("train_mse", loss, prog_bar=True, sync_dist=True)
        # self.log("batch_train_mse", mse, prog_bar=True, sync_dist=True)
        self.log("train_psnr", w_psnr_val, prog_bar=True, sync_dist=True)
        self.log("train_rmse", rmse, prog_bar=True, sync_dist=True)

        return {"loss": loss, "train_psnr": w_psnr_val.item()}


    def on_train_epoch_end(self):
        '''
        # Uncomment this if you want to visualize grid count
        grid_count = np.save('./grid_count', self.posenc.grid_count)
        grid_count = np.load('./grid_count.npy')
        n_levels = grid_count.shape[0]-1

        def visualize_and_save_healpix_maps(grid_count, n_levels):
            for i in range(n_levels):
                nside = 2**i
                npix = int(hp.nside2npix(nside))
                # For each level, visualize the grid count as a HEALPix map
                hp.mollview(grid_count[i][:npix], title=f"Level {i+1} Grid Count", cmap='viridis')
                plt.savefig(f"./image/healpix_level_{i+1}.png")  # Save the figure
                plt.close()  #       
        visualize_and_save_healpix_maps(grid_count, n_levels)
        '''
        self.eval()
        with torch.no_grad():
            # if self.current_epoch % 10 ==0:
            self.metric_all(device=self._device, mode="train")
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.log(
                "best_full_train_psnr",
                self.best_full_train_psnr,
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
            "monitor": "train_rmse",
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
