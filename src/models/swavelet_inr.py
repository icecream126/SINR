from argparse import ArgumentParser

import pytorch_lightning as pl
import torchmetrics as tm

import sys
sys.path.append('./')

import torch
from torch import nn
from math import pi
from torch.optim import lr_scheduler

from src.utils.core import parse_t_f
from src.utils import initializers as init



class SphericalGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            input_dim: Input features
            wavelet_dim; Output features
            is_first: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, wavelet_dim, omega_0=30.0, sigma_0=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma_0
        self.wavelet_dim = wavelet_dim
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1))
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1))

        self.theta_0 = nn.Parameter(torch.empty(1, wavelet_dim))
        self.phi_0 = nn.Parameter(torch.empty(1, wavelet_dim))
        nn.init.uniform_(self.theta_0, 0, 2*pi)
        nn.init.uniform_(self.phi_0, 0, 2*pi)


    def forward(self, x):
        theta = x[..., 0:1].repeat(1, 1, self.wavelet_dim)
        phi = x[..., 1:2].repeat(1, 1, self.wavelet_dim)
        time = x[..., 2:]

        tan = torch.tan((theta-self.theta_0)/2)
        cos = torch.cos(phi-self.phi_0)

        gauss = torch.exp(-torch.square(self.scale_0*tan))
        
        angle = self.omega_0 * tan * cos

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        spherical_gabor = torch.cat([gauss*cos, gauss*sin], dim=-1)

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
        # hidden_dim = 2*wavelet_dim+1
        in_dim = input_dim
        # in_dim=3
        out_dim = hidden_dim
        
        # 앞에서 layer 하나 spherical activation으로 따로 정의해줬으니까 n_layers-1까지만 돌기
        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_gabor_layer
            elif i==1:
                layer = nn.Linear(2*wavelet_dim+1, hidden_dim) #  첫 레이어에서는 input으로 (\theta,\phi,t)를 받으므로 input_dim을 3으로 설정
            else : 
                layer = nn.Linear(hidden_dim, out_dim)
            
            # layer = nn.Linear(in_dim,out_dim)


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
                act = nn.ReLU() # 첫번째 layer 이후 activation은 ReLU
                self.model.append(act)

            if i<n_layers-1:
                if bn:
                    self.model.append(nn.LayerNorm(out_dim))
                if dropout > 0:
                    self.model.append(nn.Dropout(dropout))

            in_dim = hidden_dim

            out_dim = hidden_dim
            if i + 1 == n_layers - 1:
                out_dim = output_dim

    def forward(self, x):
        for layer in self.model:
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
        wavelet_dim: int = 16,
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
        parser.add_argument("--wavelet_dim", type=int, default=16)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--lr_patience", type=int, default=1000)
        parser.add_argument("--geometric_init", type=parse_t_f, default=False)
        parser.add_argument("--beta", type=int, default=0)
        parser.add_argument("--sine", type=parse_t_f, default=False)
        parser.add_argument("--all_sine", type=parse_t_f, default=False)
        parser.add_argument("--skip", type=parse_t_f, default=True)
        parser.add_argument("--bn", type=parse_t_f, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)

        return parser