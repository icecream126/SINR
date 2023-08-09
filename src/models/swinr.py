import sys
sys.path.append('./')

import torch
import numpy as np
from math import pi
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler

from src.utils.sine import Sine
from src.utils.psnr import mse2psnr
from src.utils import initializers as init


class SphericalGaborLayer(nn.Module):
    def __init__(
            self, 
            wavelet_dim=128, 
            omega=30.0, 
            sigma=10.0, 
            time=True
        ):
        super().__init__()

        self.time = time
        self.wavelet_dim = wavelet_dim
        self.omega_0 = omega
        self.scale_0 = sigma

        self.dilate = nn.Parameter(torch.empty(1, 1, wavelet_dim))
        nn.init.normal_(self.dilate) 

        self.u = nn.Parameter(torch.empty(wavelet_dim))
        self.v = nn.Parameter(torch.empty(wavelet_dim))
        self.w = nn.Parameter(torch.empty(wavelet_dim))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)

    def forward(self, input):
        points = input[..., 0:3]

        zeros = torch.zeros(self.wavelet_dim, device=points.device)
        ones = torch.ones(self.wavelet_dim, device=points.device)

        alpha = 2*pi*self.u
        beta = torch.arccos(torch.clamp(2*self.v-1, -1+1e-6, 1-1e-6))
        gamma = 2*pi*self.w
        
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)

        Rz_alpha = torch.stack([
            torch.stack([cos_alpha, -sin_alpha, zeros], 1), 
            torch.stack([sin_alpha,  cos_alpha, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        Rx_beta = torch.stack([
            torch.stack([ ones,     zeros,      zeros], 1), 
            torch.stack([zeros, cos_beta, -sin_beta], 1), 
            torch.stack([zeros, sin_beta,  cos_beta], 1)
            ], 1)

        Rz_gamma = torch.stack([
            torch.stack([cos_gamma, -sin_gamma, zeros], 1), 
            torch.stack([sin_gamma,  cos_gamma, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        R = torch.bmm(torch.bmm(Rz_gamma, Rx_beta), Rz_alpha)

        points = torch.matmul(R, points.unsqueeze(2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        dilate = torch.exp(self.dilate)

        gauss = torch.exp(self.scale_0*self.scale_0*dilate*dilate*(z-1)/(1e-6+1+z)) 

        angle = 2 * self.omega_0 * dilate * x / (1e-6+1+z)

        real_sinusoid = torch.cos(angle)
        img_sinusoid = torch.sin(angle)

        real_gabor = gauss * real_sinusoid #/ (1e-6+1+z)
        img_gabor = gauss * img_sinusoid #/ (1e-6+1+z)

        # if self.cocycle:
        #     dilate_inv_square = 1 / torch.square(dilate)
        #     cocycle = 4 * dilate_inv_square / torch.square((dilate_inv_square-1)*z+(dilate_inv_square+1))
        #     sqrt_cocycle = torch.sqrt(cocycle)

        #     real_gabor = sqrt_cocycle * real_gabor
        #     img_gabor = sqrt_cocycle * img_gabor

        out = torch.cat([real_gabor, img_gabor], dim=-1)

        if self.time:
            time = input[..., 3:]
            out = torch.cat([out, time], dim=-1)
        return out


class MLP(nn.Module):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        wavelet_dim: int = 64, 
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
        wavelet_dim: int,
        omega: float,
        sigma: float,
        n_layers: int,
        time: bool,
        skip: bool,
        sine: bool,
        all_sine: bool,
    ):
        super().__init__()

        self.spherical_gabor_layer = SphericalGaborLayer(wavelet_dim, omega, sigma, time)

        # Modules
        self.model = nn.ModuleList()
        in_dim = hidden_dim
        out_dim = hidden_dim

        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_gabor_layer
            elif i==1:
                if time:
                    layer = nn.Linear(2*wavelet_dim+1, hidden_dim)
                else:
                    layer = nn.Linear(2*wavelet_dim, hidden_dim)
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
    

class SWINR(pl.LightningModule):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        wavelet_dim: int = 64, 
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
        wavelet_dim: int = 128,
        omega: float=30.,
        sigma: float=100.,
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
            wavelet_dim,
            omega,
            sigma,
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