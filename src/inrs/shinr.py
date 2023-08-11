import torch
import numpy as np
from torch import nn
from math import ceil

from .relu import ReLULayer
from .siren import SineLayer
from .wire import GaborLayer
from utils.spherical_harmonics import get_spherical_harmonics

hidden_inr_dict = {
    'relu': ReLULayer,
    'siren': SineLayer,
    'wire': GaborLayer,
}

class SphericalHarmonicsLayer(nn.Module):
    def __init__(
            self, 
            max_order, 
            time,
            omega,
            **kwargs,
        ):
        super().__init__()
        self.max_order = max_order
        self.hidden_dim = (max_order+1)**2
        self.time = time
        self.omega = omega

        if time:
            self.linear = nn.Linear(1, self.hidden_dim)
            with torch.no_grad():
                self.linear.weight.uniform_(-1, 1)
        
    def forward(self, input):
        theta = input[..., 0]
        phi = input[..., 1]

        sh_list = []
        for l in range(self.max_order+1):
            sh = get_spherical_harmonics(l, phi, theta)
            sh_list.append(sh)

        out = torch.cat(sh_list, dim=-1)

        if self.time:
            time = input[..., 2:]
            lin = self.linear(time)
            omega = self.omega * lin
            out = out * torch.sin(omega)
        return out

class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers,
            hidden_inr,
            max_order,
            time,
            skip=True,
            omega=10.,
            sigma=10.,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(max_order, time, omega))

        self.nonlin = hidden_inr_dict[hidden_inr]

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(self.nonlin((max_order+1)**2,
                                            hidden_dim,
                                            is_first=False,
                                            omega=omega,
                                            sigma=sigma))     
            else:
                if skip and i == ceil(hidden_layers/2):
                    self.net.append(self.nonlin(hidden_dim+input_dim,
                                                hidden_dim,
                                                is_first=False,
                                                omega=omega,
                                                sigma=sigma))
                else:
                    self.net.append(self.nonlin(hidden_dim,
                                                hidden_dim,
                                                is_first=False,
                                                omega=omega,
                                                sigma=sigma))

        final_linear = nn.Linear(hidden_dim, output_dim) 

        if hidden_inr == 'siren':
            with torch.no_grad():
                const = np.sqrt(6/hidden_dim)/max(omega, 1e-12)
                final_linear.weight.uniform_(-const, const)
        
        self.net.append(final_linear)
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers/2)+1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x