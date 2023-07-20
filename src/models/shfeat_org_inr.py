from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchmetrics as tm

import sys
sys.path.append('./')
from src.models.core import parse_t_f
from torch import nn
from torch.optim import lr_scheduler
import numpy as np
import torch

from src.models import initializers as init
from scipy.special import lpmv
import math


class SphericalHarmonicsLayer(nn.Module):
    def __init__(self, max_order, hidden_dim):
        super(SphericalHarmonicsLayer, self).__init__()
        self.max_order = max_order

        self.theta_layer = nn.Linear(1, hidden_dim)
        self.phi_layer = nn.Linear(1, hidden_dim)
        self.time_layer = nn.Linear(1, hidden_dim)
        
    def forward(self, x):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        theta = x[:,:,0].unsqueeze(dim=2) # torch.Size([3,5000,1])
        phi = x[:,:,1].unsqueeze(dim=2) # torch.Size([3,5000,1])
        time = x[:,:,2].unsqueeze(dim=2) # torch.Size([3,5000,1])

        theta = self.theta_layer(theta) # torch.Size([3,5000, k])
        phi = self.phi_layer(phi) # torch.Size([3,5000, k])
        time = self.time_layer(time) # torch.Size([3,5000,k]) # 굳이 할 필요는 없지만 나중에 concat 해주기 위해 dimension 맞춰주는 용도

        theta = theta.reshape(batch_size,-1,1) # torch.Size([3,5000*k,1])
        phi = phi.reshape(batch_size,-1,1) # torch.Size([3,5000*k,1])
        time = time.reshape(batch_size, -1, 1).cuda() # torch.Size(3, 5000*k,1)

        theta = theta.detach().cpu().numpy()
        phi = phi.detach().cpu().numpy()

        y = []
        for l in range(self.max_order + 1):
            for m in range(-l, l + 1):
                Klm = math.sqrt((2*l+1) * math.factorial(l-m) / (4*math.pi * math.factorial(l+m)))
                
                if m > 0:
                    Ylm = Klm * math.sqrt(2) * lpmv(m, l, np.cos(theta)) * np.cos(m * phi)
                elif m == 0:
                    Ylm = Klm * lpmv(0, l, np.cos(theta))
                else:
                    Ylm = Klm * math.sqrt(2) * lpmv(-m, l, np.cos(theta)) * np.sin(-m * phi)
                #  Ylm = torch.Size([3,5000*k,1])
                y.append(torch.Tensor(Ylm).cuda())
        y = torch.stack(y, dim=2).squeeze(-1) # torch.Size([3, 5000*k, (order+1)**2])
        
        x = torch.cat([y, time], dim=-1) # torch.Size([3,5000*k,(max_order+1)**2+1])
        x = x.reshape(batch_size, num_samples,-1) # torch.Size([3,5000, ((max_order+1)**2+1)*hidden_dim])

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
        self.max_order = input_dim-1 # change this for spherical harmonics embedding max order (input_dim = dataset.n_fourier+1)
        print('Spherical Embedding order : ',self.max_order)
        self.spherical_harmonics_layer = SphericalHarmonicsLayer(self.max_order, hidden_dim)

        # Modules
        self.model = nn.ModuleList()
        # hidden_dim = (self.max_order+1)**2
        in_dim = input_dim
        # in_dim=3
        out_dim = hidden_dim
        
        # 앞에서 layer 하나 spherical activation으로 따로 정의해줬으니까 n_layers-1까지만 돌기
        for i in range(n_layers):   
            
            if i==0:
                layer = self.spherical_harmonics_layer
            elif i==1:
                layer = nn.Linear(((self.max_order+1)**2+1)*hidden_dim,hidden_dim) #  첫 레이어에서는 input으로 (\theta,\phi,t)를 받으므로 input_dim을 3으로 설정
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
        x_in = x    
        for i, layer in enumerate(self.model):
            x = layer(x)
        return x




class SHFeatINR(pl.LightningModule):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        dataset_size: int, number of samples in the dataset (for autodecoder latents)
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        lr: float = 0.0005, learning rate
        lr_patience: int = 500, learning rate patience (in number of epochs)
        latents: bool = False, make the model an autodecoder with learnable latents
        latent_dim: int = 256, size of the latents
        lambda_latent: float = 0.0001, regularization factor for the latents
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
        hidden_dim: int = 64, # 원래 512
        n_layers: int = 4,
        lr: float = 0.0005,
        lr_patience: int = 500,
        latents: bool = False,
        latent_dim: int = 256,
        lambda_latent: float = 0.0001,
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
        self.use_latents = latents
        self.latent_dim = latent_dim
        self.lambda_latent = lambda_latent
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = True
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.classifier = classifier

        self.sync_dist = torch.cuda.device_count() > 1

        # Compute true input dimension
        input_dim_true = input_dim
        if latents:
            input_dim_true += latent_dim

        # Modules
        self.model = MLP(
            input_dim_true,
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

        # Latent codes
        if latents:
            self.latents = nn.Embedding(dataset_size, latent_dim)

        # Loss
        if self.classifier:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Metrics
        self.r2_score = tm.R2Score(output_dim)

        self.save_hyperparameters()

    def forward(self, points):
        return self.model(points)

    def forward_with_preprocessing(self, data):
        points, indices = data
        if self.use_latents:
            latents = self.latents(indices)
            points = self.add_latent(points, latents)
        return self.forward(points)

    def add_latent(self, points, latents):
        n_points = points.shape[1]
        latents = latents.unsqueeze(1).repeat(1, n_points, 1)

        return torch.cat([latents, points], dim=-1)

    def latent_size_reg(self, indices):
        latent_loss = self.latents(indices).norm(dim=-1).mean()
        return latent_loss

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

    def training_step(self, data):
        inputs, target, indices = data["inputs"], data["target"], data["index"]
        # print('inputs shape : ',inputs.shape) # torch.Size([5, 5000, 35])
        # print('inputs : ',inputs)
        # print('target shape : ',target.shape) # torch.Size([5, 5000, 1])
        # print('target : ',target)
        # print('index shape : ',indices.shape)
        # print('index : ',index)

        # Add latent codes
        if self.use_latents:
            latents = self.latents(indices)
            inputs = self.add_latent(inputs, latents)

        inputs.requires_grad_()

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        if self.classifier:
            pred = torch.permute(pred, (0, 2, 1))

        main_loss = self.loss_fn(pred, target)
        self.log("main_loss", main_loss, prog_bar=True, sync_dist=self.sync_dist)
        loss = main_loss

        if not self.classifier:
            self.r2_score(
                pred.view(-1, self.output_dim), target.view(-1, self.output_dim)
            )
            self.log(
                "r2_score", self.r2_score, prog_bar=True, on_epoch=True, on_step=False
            )

        # Latent size regularization
        if self.use_latents:
            latent_loss = self.latent_size_reg(indices)
            loss += self.lambda_latent * latent_loss
            self.log(
                "latent_loss", latent_loss, prog_bar=True, sync_dist=self.sync_dist
            )

        self.log("loss", loss, sync_dist=self.sync_dist)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )
        sch_dict = {"scheduler": scheduler, "monitor": "loss", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": sch_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--hidden_dim", type=int, default=64) # 원래 512
        parser.add_argument("--n_layers", type=int, default=4)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--lr_patience", type=int, default=1000)
        parser.add_argument("--latents", action="store_true")
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--lambda_latent", type=float, default=0.0001)
        parser.add_argument("--geometric_init", type=parse_t_f, default=False)
        parser.add_argument("--beta", type=int, default=0)
        parser.add_argument("--sine", type=parse_t_f, default=False)
        parser.add_argument("--all_sine", type=parse_t_f, default=False)
        parser.add_argument("--skip", type=parse_t_f, default=True)
        parser.add_argument("--bn", type=parse_t_f, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)

        return parser
