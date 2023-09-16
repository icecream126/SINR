import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import torch
from torch import nn
from math import pi, ceil

import numpy as np
import torch

# from src.models import initializers as init
# from src.modules.sine import Sine
from torch import nn
import torchmetrics as tm

from .model import MODEL, DENOISING_MODEL
from .relu import ReLULayer
from utils.utils import (
    to_cartesian,
    mse2psnr,
    Sine,
    geometric_initializer,
    first_layer_sine_initializer,
    sine_initializer,
)


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

        # Modules
        self.model = nn.ModuleList()
        in_dim = input_dim
        out_dim = hidden_dim
        for i in range(n_layers):
            layer = nn.Linear(in_dim, out_dim)

            # Custom initializations
            if geometric_init:
                if i == n_layers - 1:
                    geometric_initializer(layer, in_dim)
            elif sine:
                if i == 0:
                    first_layer_sine_initializer(layer)
                elif all_sine:
                    sine_initializer(layer)

            self.model.append(layer)

            # Activation, BN, and dropout
            if i < n_layers - 1:
                if sine:
                    if i == 0:
                        act = Sine()
                    else:
                        act = Sine() if all_sine else nn.Tanh()
                elif beta > 0:
                    act = nn.Softplus(beta=beta)  # IGR uses Softplus with beta=100
                else:
                    act = nn.ReLU(inplace=True)
                self.model.append(act)
                if bn:
                    self.model.append(nn.LayerNorm(out_dim))
                if dropout > 0:
                    self.model.append(nn.Dropout(dropout))

            in_dim = hidden_dim
            # Skip connection
            if i + 1 == int(np.ceil(n_layers / 2)) and skip:
                self.skip_at = len(self.model)
                in_dim += input_dim

            out_dim = hidden_dim
            if i + 1 == n_layers - 1:
                out_dim = output_dim

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.model):
            if i == self.skip_at:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)

        return x


class INR(MODEL):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        geometric_init: bool = False,
        beta: int = 0,
        sine: bool = False,
        all_sine: bool = False,
        skip: bool = True,
        bn: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout

        # Modules
        self.model = MLP(
            input_dim,
            output_dim,
            hidden_dim,
            n_layers,
            geometric_init,
            beta,
            sine,
            all_sine,
            skip,
            bn,
            dropout,
        )

        # Loss
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, points):
        return self.model(points)

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

        # Predict signal
        pred = self.forward(inputs)

        # import pdb

        # pdb.set_trace()

        # Loss
        main_loss = self.loss_fn(pred, target)
        self.log("main_loss", main_loss, prog_bar=True, sync_dist=True)
        loss = main_loss

        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())

        self.log("batch_train_mse", loss, sync_dist=True)
        self.log(
            "batch_train_psnr",
            w_psnr_val,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        main_loss = self.loss_fn(pred, target)
        self.log("main_loss", main_loss, prog_bar=True, sync_dist=True)
        loss = main_loss

        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())

        self.log("batch_valid_mse", loss, sync_dist=True)
        self.log("batch_valid_psnr", w_psnr_val)

        return {"batch_valid_mse": loss.item()}

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        main_loss = self.loss_fn(pred, target)
        self.log("main_loss", main_loss, prog_bar=True, sync_dist=True)
        loss = main_loss

        w_psnr_val = mse2psnr(loss.detach().cpu().numpy())

        self.log("batch_test_mse", loss, sync_dist=True)
        self.log("batch_test_psnr", w_psnr_val)

        return {"batch_test_mse": loss.item(), "batch_test_psnr": w_psnr_val.item()}


class DENOISING_INR(DENOISING_MODEL):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_layers,
        levels,
        time,
        skip,
        omega,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        # self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(levels, time, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(self.nonlin(levels**2, hidden_dim))
            elif skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + input_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x
