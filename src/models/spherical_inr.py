from argparse import ArgumentParser

import pytorch_lightning as pl
import torchmetrics as tm

import sys
sys.path.append('./')

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler

from src.utils import initializers as init
from src.utils.spherical_harmonics import get_spherical_harmonics

class SphericalHarmonicsLayer(nn.Module):
    def __init__(self, max_order):
        super(SphericalHarmonicsLayer, self).__init__()
        self.max_order = max_order

        # self.theta_layer = nn.Linear(1, hidden_dim)
        # self.phi_layer = nn.Linear(1, hidden_dim)
        # self.time_layer = nn.Linear(1, hidden_dim)
        
    def forward(self, x):

        # batch_size = x.shape[0]
        # num_samples = x.shape[1]

        theta = x[:,:,0].unsqueeze(-1) # torch.Size([3,5000,1])
        phi = x[:,:,1].unsqueeze(-1) # torch.Size([3,5000,1])
        time = x[:,:,2].unsqueeze(-1) # torch.Size([3,5000,1])

        # theta = self.theta_layer(theta) # torch.Size([3,5000, k])
        # phi = self.phi_layer(phi) # torch.Size([3,5000, k])
        # time = self.time_layer(time) # torch.Size([3,5000,k]) # 굳이 할 필요는 없지만 나중에 concat 해주기 위해 dimension 맞춰주는 용도

        # theta = theta.reshape(batch_size, -1) # torch.Size([3,5000*k,1])
        # phi = phi.reshape(batch_size, -1) # torch.Size([3,5000*k,1])
        # time = time.reshape(batch_size, -1) # torch.Size(3, 5000*k,1)

        # spherical harmonics using pytorch
        sh_list = []
        for l in range(self.max_order+1):
            sh = get_spherical_harmonics(l, phi, theta)
            sh_list.append(sh)

        #clear_spherical_harmonics_cache()
        sh = torch.cat(sh_list, dim=-1)

        x = torch.cat([sh, time], dim=-1) # torch.Size([3,5000*k,(max_order+1)**2+1])
        # x = x.reshape(batch_size, num_samples,-1) # torch.Size([3,5000, ((max_order+1)**2+1)*hidden_dim])
        
        # currently, without detach, gradient explodes...
        # x = x.detach()
        
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
        max_order: int,
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
        self.max_order = max_order

        self.spherical_harmonics_layer = SphericalHarmonicsLayer(self.max_order)

        # Modules
        self.model = nn.ModuleList()
        in_dim = hidden_dim
        out_dim = hidden_dim

        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_harmonics_layer
            elif i==1:
                layer = nn.Linear((self.max_order+1)**2+1, hidden_dim)
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




class SphericalINR(pl.LightningModule):
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
        max_order: int = 3,
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
            max_order,
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
        parser.add_argument("--max_order", type=int, default=3)
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