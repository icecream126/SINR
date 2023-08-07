import sys
sys.path.append('./')

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler


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
    
    def __init__(self, input_dim, output_dim, omega0=10.0, sigma0=10.0):
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
    
    
class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            input_dim: Input features
            output_dim; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        is_first=False, 
        omega0=30.0, 
        sigma0=10.0
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.input_dim = input_dim
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        
        self.linear = nn.Linear(input_dim,
                                output_dim,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    

class WIRE(pl.LightningModule):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        first_omega_0=30., 
        hidden_omega_0=30., 
        scale=10.0,
        lr: float = 0.0005,
        lr_patience: int = 500,
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.lr_patience = lr_patience
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_dim = int(hidden_dim/np.sqrt(2))
        dtype = torch.cfloat
        # self.complex = True
        # self.wavelet = 'gabor'    
        
        # Legacy parameter
        # self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(input_dim,
                                    hidden_dim, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True))

        for i in range(n_layers):
            self.net.append(self.nonlin(hidden_dim,
                                        hidden_dim, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_dim,
                                 output_dim,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)

        self.sync_dist = torch.cuda.device_count() > 1

        self.loss_fn = nn.MSELoss()
        self.min_valid_loss = None
    
    def forward(self, coords):
        output = self.net(coords)
        return output.real

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        return loss
    
    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        pred = self.forward(inputs)

        loss = self.loss_fn(pred, target)
        self.log("test_mse", loss)
        self.log("min_valid_loss", self.min_valid_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {"scheduler": scheduler, "monitor": 'valid_loss', "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}