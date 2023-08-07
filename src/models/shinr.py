import sys
sys.path.append('./')

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler

from src.utils.sine import Sine
from src.utils.psnr import mse2psnr
from src.utils import initializers as init
from src.utils.spherical_harmonics import get_spherical_harmonics

class SphericalHarmonicsLayer(nn.Module):
    def __init__(self, max_order, time):
        super(SphericalHarmonicsLayer, self).__init__()
        self.max_order = max_order
        self.time = time
        
    def forward(self, input):
        theta = input[..., 0].unsqueeze(-1) 
        phi = input[..., 1].unsqueeze(-1) 

        sh_list = []
        for l in range(self.max_order+1):
            sh = get_spherical_harmonics(l, phi, theta)
            sh_list.append(sh)

        out = torch.cat(sh_list, dim=-1)

        if self.time:
            time = input[..., 2].unsqueeze(-1) 
            out = torch.cat([out, time], dim=-1)  
        return out


class MLP(nn.Module):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        max_order: int = 3, 
        n_layers: int = 4, number of layers (total, including first and last)
        time: bool = True,
        skip: bool = True, add a skip connection to the middle layer
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        max_order: int,
        n_layers: int,
        time: bool,
        skip: bool,
        sine: bool,
        all_sine: bool,
    ):
        super().__init__()

        self.spherical_harmonics_layer = SphericalHarmonicsLayer(max_order, time)

        # Modules
        self.model = nn.ModuleList()
        in_dim = hidden_dim
        out_dim = hidden_dim

        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_harmonics_layer
            elif i==1:
                if time:
                    layer = nn.Linear((self.max_order+1)**2+1, hidden_dim)
                else:
                    layer = nn.Linear((self.max_order+1)**2, hidden_dim)
            else : 
                layer = nn.Linear(in_dim, out_dim)

            # Custom initializations
            if sine:
                if i == 0:
                    init.first_layer_sine_initializer(layer)
                elif all_sine:
                    init.sine_initializer(layer)

            self.model.append(layer)

            if i < n_layers - 1:
                if sine:
                    if i == 0:
                        act = Sine()
                    else:
                        act = Sine() if all_sine else nn.Tanh()
                else:
                    act = nn.ReLU(inplace=True)
                self.model.append(act)

            in_dim = hidden_dim

            # Skip connection
            if i + 1 == int(np.ceil(n_layers / 2)) and skip:
                self.skip_at = len(self.model)
                in_dim += input_dim

            if i + 1 == n_layers - 1:
                out_dim = output_dim

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.model):
            if i == self.skip_at:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x


class SHINR(pl.LightningModule):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        max_order: int = 3, 
        n_layers: int = 4, number of layers (total, including first and last)
        time: bool = True,
        skip: bool = True, add a skip connection to the middle layer
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
        lr: float = 0.0005, learning rate
        lr_patience: int = 500, learning rate patience (in number of epochs)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        max_order: int = 3,
        n_layers: int = 8,
        time: bool = True,
        skip: bool = True,
        sine: bool = False,
        all_sine: bool = False,
        lr: float = 0.001,
        lr_patience: int = 1000,
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience

        self.sync_dist = torch.cuda.device_count() > 1

        # Modules
        self.model = MLP(
            input_dim,
            output_dim,
            hidden_dim,
            max_order,
            n_layers,
            time,
            skip,
            sine,
            all_sine,
        )

        self.loss_fn = nn.MSELoss()
        self.min_valid_loss = None

    def forward(self, points):
        return self.model(points)

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train_psnr", mse2psnr(loss), prog_bar=True, sync_dist=self.sync_dist)
        return loss
    
    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("valid_psnr", mse2psnr(loss), prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("test_mse", loss)
        self.log("test_psnr", mse2psnr(loss), prog_bar=True, sync_dist=self.sync_dist)
        self.log("min_valid_loss", self.min_valid_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {"scheduler": scheduler, "monitor": 'valid_loss', "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}