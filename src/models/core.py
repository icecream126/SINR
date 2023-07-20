import numpy as np
import torch

import sys
sys.path.append('./')
from src.models import initializers as init
from src.modules.sine import Sine
from torch import nn
import math


class SphericalHarmonicsLayer(nn.Module):
    def __init__(self, max_order,hidden_dim):
        super(SphericalHarmonicsLayer, self).__init__()
        self.max_order = max_order

        self.theta_layer = nn.Linear(1, hidden_dim)
        self.phi_layer = nn.Linear(1, hidden_dim)
        self.time_layer = nn.Linear(1, hidden_dim)

    def lpmv(l, m, x):
        """Associated Legendre function including Condon-Shortley phase.

        Args:
            m: int order 
            l: int degree
            x: float argument tensor
        Returns:
            tensor of x-shape
        """
        # Check memoized versions
        m_abs = abs(m)

        if m_abs > l:
            return None

        if l == 0:
            return torch.ones_like(x)
        
        # Check if on boundary else recurse solution down to boundary
        if m_abs == l:
            # Compute P_m^m
            y = (-1)**m_abs * semifactorial(2*m_abs-1)
            y *= torch.pow(1-x*x, m_abs/2)
            return negative_lpmv(l, m, y)

        # Recursively precompute lower degree harmonics
        lpmv(l-1, m, x)

        # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
        # Inplace speedup
        y = ((2*l-1) / (l-m_abs)) * x * lpmv(l-1, m_abs, x)

        if l - m_abs > 1:
            y -= ((l+m_abs-1)/(l-m_abs)) * CACHE[(l-2, m_abs)]
        
        if m < 0:
            y = self.negative_lpmv(l, m, y)
        return y

        
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

        y = []
        for l in range(self.max_order + 1):
            for m in range(-l, l + 1):
                Klm = math.sqrt((2*l+1) * math.factorial(l-m) / (4*math.pi * math.factorial(l+m)))
                
                if m > 0:
                    Ylm = Klm * math.sqrt(2) * lpmv(m, l, torch.cos(theta)) * torch.cos(m * phi)
                elif m == 0:
                    Ylm = Klm * lpmv(0, l, torch.cos(theta))
                else:
                    Ylm = Klm * math.sqrt(2) * lpmv(-m, l, torch.cos(theta)) * torch.sin(-m * phi)
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


def parse_t_f(arg):
    """Used to create flags like --flag=True with argparse"""
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError("Arg must be True or False")
