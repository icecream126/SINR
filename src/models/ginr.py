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
        hidden_layers: int = 4,
        geometric_init: bool = False,
        beta: int = 0,
        sine: bool = False,
        all_sine: bool = False,
        skip: bool = True,
        bn: bool = False,
        dropout: float = 0.0,
        latent_dim: int = 256,
        time: bool = False,  # True when using temporal data / default: [1] to [256] embedding
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = hidden_layers
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.time = time

        # Compute true input dimension
        input_dim_true = input_dim
        if time:
            input_dim_true += latent_dim - 1

        # Modules
        self.model = MLP(
            input_dim_true,
            output_dim,
            hidden_dim,
            hidden_layers,
            geometric_init,
            beta,
            sine,
            all_sine,
            skip,
            bn,
            dropout,
        )
        # Latent codes
        if time:
            self.latents = nn.Linear(1, latent_dim)

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

    def add_latent(self, points, latents):
        return torch.cat([latents, points], dim=-1)

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        if self.time:
            time = data["time"]
            latents = self.latents(time)
            inputs = self.add_latent(inputs, latents)

        # Predict signal
        pred = self.forward(inputs)

        # import pdb

        # pdb.set_trace()

        # Loss
        main_loss = self.loss_fn(pred, target)
        loss = main_loss

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(pred)
            target = self.scaler.inverse_transform(target)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            mse = mse.mean()
        else:
            mse = loss

        w_psnr_val = mse2psnr(mse.detach().cpu().numpy())

        self.log("batch_train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_train_mse", mse, prog_bar=True, sync_dist=True)
        self.log(
            "batch_train_psnr",
            w_psnr_val,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]

        # import pdb

        # pdb.set_trace()
        if self.time:
            time = data["time"]
            latents = self.latents(time)
            inputs = self.add_latent(inputs, latents)

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        main_loss = self.loss_fn(pred, target)
        loss = main_loss

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(pred)
            target = self.scaler.inverse_transform(target)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            mse = mse.mean()
        else:
            mse = loss

        w_psnr_val = mse2psnr(mse.detach().cpu().numpy())

        self.log("batch_valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_valid_mse", mse, prog_bar=True, sync_dist=True)
        self.log("batch_valid_psnr", w_psnr_val, prog_bar=True, sync_dist=True)

        return {"batch_valid_mse": loss.item()}

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        if self.time:
            time = data["time"]
            latents = self.latents(time)
            inputs = self.add_latent(inputs, latents)

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        main_loss = self.loss_fn(pred, target)
        loss = main_loss

        if self.normalize:
            self.scaler.match_device(pred)
            pred = self.scaler.inverse_transform(pred)
            target = self.scaler.inverse_transform(target)
            mse = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
            mse = mse.mean()
        else:
            mse = loss

        w_psnr_val = mse2psnr(mse.detach().cpu().numpy())

        self.log("batch_test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("batch_test_mse", mse, prog_bar=True, sync_dist=True)
        self.log("batch_test_psnr", w_psnr_val, prog_bar=True, sync_dist=True)

        return {"batch_test_mse": loss.item(), "batch_test_psnr": w_psnr_val.item()}
