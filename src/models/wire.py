import sys
sys.path.append('./')

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler

from src.utils.psnr import mse2psnr


class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            input_dim: Input features
            output_dim; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, input_dim, output_dim, omega0=30.0, sigma0=10.0):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        
        self.input_dim = input_dim
        
        self.freqs = nn.Linear(input_dim, output_dim)
        self.scale = nn.Linear(input_dim, output_dim)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))
    
    
# class ComplexGaborLayer(nn.Module):
#     '''
#         Implicit representation with complex Gabor nonlinearity
        
#         Inputs;
#             input_dim: Input features
#             output_dim; Output features
#             bias: if True, enable bias for the linear operation
#             is_first: Legacy SIREN parameter
#             omega_0: Legacy SIREN parameter
#             omega0: Frequency of Gabor sinusoid term
#             sigma0: Scaling of Gabor Gaussian term
#             trainable: If True, omega and sigma are trainable parameters
#     '''
    
#     def __init__(
#         self, 
#         input_dim, 
#         output_dim, 
#         is_first=False, 
#         omega0=30.0, 
#         sigma0=10.0
#     ):
#         super().__init__()
#         self.omega_0 = omega0
#         self.scale_0 = sigma0
#         self.is_first = is_first
        
#         self.input_dim = input_dim
        
#         if self.is_first:
#             dtype = torch.float
#         else:
#             dtype = torch.cfloat
            
#         # Set trainable parameters if they are to be simultaneously optimized
        
#         self.linear = nn.Linear(input_dim,
#                                 output_dim,
#                                 dtype=dtype)
    
#     def forward(self, input):
#         lin = self.linear(input)
#         omega = self.omega_0 * lin
#         scale = self.scale_0 * lin
        
#         return torch.exp(1j*omega - scale.abs().square())
    

class MLP(nn.Module):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        skip: bool = True, add a skip connection to the middle layer
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        skip: bool,
    ):
        super().__init__()

        # Modules
        self.model = nn.ModuleList()
        in_dim = input_dim
        out_dim = hidden_dim

        for i in range(n_layers):   
            layer = RealGaborLayer(in_dim, out_dim)

            self.model.append(layer)

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
    

class WIRE(pl.LightningModule):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 8,
        skip: bool = True,
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
            n_layers,
            skip
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