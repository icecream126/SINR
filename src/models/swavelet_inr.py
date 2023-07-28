from argparse import ArgumentParser

import pytorch_lightning as pl
import torchmetrics as tm

import sys
sys.path.append('./')

import numpy as np
import torch
from torch import nn
from math import pi
from torch.optim import lr_scheduler

from src.utils import initializers as init



class SphericalGaborLayer(nn.Module):
    '''
        Implicit representation with spherical Gabor nonlinearity
        
        Inputs;
            wavelet_dim; Output features
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, wavelet_dim, omega_0=60.0, sigma_0=100.0, trainable=False):
        super().__init__() 

        self.wavelet_dim = wavelet_dim
        self.omega_0 = omega_0
        self.scale_0 = sigma_0

        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)

        self.alpha = nn.Parameter(torch.empty(wavelet_dim))
        self.beta = nn.Parameter(torch.empty(wavelet_dim))
        self.gamma = nn.Parameter(torch.empty(wavelet_dim))
        nn.init.uniform_(self.alpha, 0, 2*pi)
        nn.init.uniform_(self.beta, 0, pi)
        nn.init.uniform_(self.gamma, 0, 2*pi)

    def forward(self, input):
        points = input[..., 0:3]
        time = input[..., 3:4]

        # generate euler matrix
        zeros = torch.zeros(self.wavelet_dim, device=input.device)
        ones = torch.ones(self.wavelet_dim, device=input.device)

        cos_alpha = torch.cos(self.alpha)
        cos_beta = torch.cos(self.beta)
        cos_gamma = torch.cos(self.gamma)
        sin_alpha = torch.sin(self.alpha)
        sin_beta = torch.sin(self.beta)
        sin_gamma = torch.sin(self.gamma)

        alpha_row_1 = torch.stack([cos_alpha, -sin_alpha, zeros], 1)
        alpha_row_2 = torch.stack([sin_alpha, cos_alpha, zeros], 1)
        alpha_row_3 = torch.stack([zeros, zeros, ones], 1)
        yaw = torch.stack([alpha_row_1, alpha_row_2, alpha_row_3], 1)

        beta_row_1 = torch.stack([ones, zeros, zeros], 1)
        beta_row_2 = torch.stack([zeros, cos_beta, -sin_beta], 1)
        beta_row_3 = torch.stack([zeros, sin_beta, cos_beta], 1)
        pitch = torch.stack([beta_row_1, beta_row_2, beta_row_3], 1)

        gamma_row_1 = torch.stack([cos_gamma, -sin_gamma, zeros], 1)
        gamma_row_2 = torch.stack([sin_gamma, cos_gamma, zeros], 1)
        gamma_row_3 = torch.stack([zeros, zeros, ones], 1)
        roll = torch.stack([gamma_row_1, gamma_row_2, gamma_row_3], 1)

        points = torch.matmul(yaw, points.unsqueeze(2).unsqueeze(-1)) 
        points = torch.matmul(pitch, points)
        points = torch.matmul(roll, points)
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        gauss = torch.exp(self.scale_0*torch.div(z-1, 1e-6+1+z))
        
        angle = self.omega_0 * torch.div(x, 1e-6+1+z)

        real_sinusoid = torch.cos(angle)
        img_sinusoid = torch.sin(angle)

        real_gabor = gauss * real_sinusoid
        img_gabor = gauss * img_sinusoid

        spherical_gabor = torch.cat([real_gabor, img_gabor], dim=-1)

        x = torch.cat([spherical_gabor, time], dim=-1)
        return x


class MLP(nn.Module):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        geometric_init: bool = False, initialize weights so that output is spherical
        beta: int = 0, if positive, use SoftPlus(beta) instead of ReLU activations
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
        skip: bool = True, add a skip connection to the middle layer
        bn: bool = False, use batch normalization
        dropout: float = 0.0, dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        wavelet_dim: int,
        geometric_init: bool,
        beta: int,
        sine: bool,
        all_sine: bool,
        skip: bool,
        bn: bool,
        dropout: float,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.wavelet_dim = wavelet_dim

        self.spherical_gabor_layer = SphericalGaborLayer(self.wavelet_dim)

        # Modules
        self.model = nn.ModuleList()
        in_dim = hidden_dim
        out_dim = hidden_dim

        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_gabor_layer
            elif i==1:
                layer = nn.Linear(2*wavelet_dim+1, hidden_dim)
            else : 
                layer = nn.Linear(in_dim, out_dim)

            # Custom initializations
            if geometric_init:
                if i == n_layers - 1:
                    init.geometric_initializer(layer, in_dim)
            elif sine:
                if i == 0:
                    init.first_layer_sine_initializer(layer)
                elif all_sine:
                    init.sine_initializer(layer)

            self.model.append(layer)

            if i<n_layers-1 and i>0:
                act = nn.ReLU() 
                self.model.append(act)

            if i<n_layers-1:
                if bn:
                    self.model.append(nn.LayerNorm(out_dim))
                if dropout > 0:
                    self.model.append(nn.Dropout(dropout))

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

    

class SwaveletINR(pl.LightningModule):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        dataset_size: int, number of samples in the dataset
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        lr: float = 0.0005, learning rate
        lr_patience: int = 500, learning rate patience (in number of epochs)
        geometric_init: bool = False, initialize weights so that output is spherical
        beta: int = 0, if positive, use SoftPlus(beta) instead of ReLU activations
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
        skip: bool = True, add a skip connection to the middle layer
        bn: bool = False, use batch normalization
        dropout: float = 0.0, dropout rate
        classifier: bool = False, use CrossEntropyLoss as loss
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dataset_size: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        wavelet_dim: int = 256,
        lr: float = 0.0005,
        lr_patience: int = 500,
        geometric_init: bool = False,
        beta: int = 0,
        sine: bool = False,
        all_sine: bool = False,
        skip: bool = True,
        bn: bool = False,
        dropout: float = 0.0,
        classifier: bool = False,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset_size = dataset_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.lr_patience = lr_patience
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = True
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.classifier = classifier

        self.sync_dist = torch.cuda.device_count() > 1

        # Modules
        self.model = MLP(
            input_dim,
            output_dim,
            hidden_dim,
            n_layers,
            wavelet_dim,
            geometric_init,
            beta,
            sine,
            all_sine,
            skip,
            bn,
            dropout,
        )

        # Loss
        if self.classifier:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Metrics
        self.train_r2_score = tm.R2Score(output_dim)
        self.valid_r2_score = tm.R2Score(output_dim)
        self.test_r2_score = tm.R2Score(output_dim)

        self.save_hyperparameters()

    def forward(self, points):
        return self.model(points)

    def forward_with_preprocessing(self, data):
        points, indices = data
        return self.forward(points)

    @staticmethod
    def gradient(inputs, outputs):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][..., -3:]
        return points_grad

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        inputs.requires_grad_()

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        if self.classifier:
            pred = torch.permute(pred, (0, 2, 1))

        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)

        if not self.classifier:
            self.train_r2_score(
                pred.view(-1, self.output_dim), target.view(-1, self.output_dim)
            )
            self.log(
                "train_r2_score", self.train_r2_score, prog_bar=True, on_epoch=True, on_step=False
            )

        return loss
    
    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        if self.classifier:
            pred = torch.permute(pred, (0, 2, 1))

        loss = self.loss_fn(pred, target)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=self.sync_dist)

        if not self.classifier:
            self.valid_r2_score(
                pred.view(-1, self.output_dim), target.view(-1, self.output_dim)
            )
            self.log(
                "valid_r2_score", self.valid_r2_score, prog_bar=True, on_epoch=True, on_step=False
            )

        return loss

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        if self.classifier:
            pred = torch.permute(pred, (0, 2, 1))

        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss)

        if not self.classifier:
            self.test_r2_score(
                pred.view(-1, self.output_dim), target.view(-1, self.output_dim)
            )
            self.log(
                "test_r2_score", self.test_r2_score
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )
        sch_dict = {"scheduler": scheduler, "monitor": "train_loss", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": sch_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--n_layers", type=int, default=4)
        parser.add_argument("--wavelet_dim", type=int, default=256)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--lr_patience", type=int, default=1000)
        parser.add_argument("--geometric_init", type=bool, default=False)
        parser.add_argument("--beta", type=int, default=0)
        parser.add_argument("--sine", type=bool, default=False)
        parser.add_argument("--all_sine", type=bool, default=False)
        parser.add_argument("--skip", type=bool, default=True)
        parser.add_argument("--bn", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)

        return parser