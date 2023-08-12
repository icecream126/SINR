import torch
import numpy as np
from torch import nn
from math import pi, ceil

from .relu import ReLULayer
from .siren import SineLayer
from .wire import GaborLayer

hidden_inr_dict = {
    'relu': ReLULayer,
    'siren': SineLayer,
    'wire': GaborLayer,
}

class SphericalGaborLayer(nn.Module):
    def __init__(
            self, 
            out_dim,
            time,
            omega,
            sigma,
            **kwargs,
        ):
        super().__init__()

        self.time = time
        self.omega = omega
        self.sigma = sigma
        self.out_dim = out_dim

        self.dilate = nn.Parameter(torch.empty(1, out_dim))
        nn.init.normal_(self.dilate)

        self.u = nn.Parameter(torch.empty(out_dim))
        self.v = nn.Parameter(torch.empty(out_dim))
        self.w = nn.Parameter(torch.empty(out_dim))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)

        if time:
            self.linear = nn.Linear(1, out_dim)

    def forward(self, input):
        points = input[..., 0:3]

        zeros = torch.zeros(self.out_dim, device=points.device)
        ones = torch.ones(self.out_dim, device=points.device)

        alpha = 2*pi*self.u
        beta = torch.arccos(torch.clamp(2*self.v-1, -1+1e-6, 1-1e-6))
        gamma = 2*pi*self.w
        
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)

        Rz_alpha = torch.stack([
            torch.stack([cos_alpha, -sin_alpha, zeros], 1), 
            torch.stack([sin_alpha,  cos_alpha, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        Rx_beta = torch.stack([
            torch.stack([ ones,     zeros,      zeros], 1), 
            torch.stack([zeros, cos_beta, -sin_beta], 1), 
            torch.stack([zeros, sin_beta,  cos_beta], 1)
            ], 1)

        Rz_gamma = torch.stack([
            torch.stack([cos_gamma, -sin_gamma, zeros], 1), 
            torch.stack([sin_gamma,  cos_gamma, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        R = torch.bmm(torch.bmm(Rz_gamma, Rx_beta), Rz_alpha)

        points = torch.matmul(R, points.unsqueeze(-2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        dilate = torch.exp(self.dilate)

        arg = 4 * dilate * dilate * (1-z) / (1e-6+1+z)

        freq_term = torch.exp(1j*2*self.omega*dilate*x/(1e-6+1+z))
        gauss_term = torch.exp(-self.sigma*self.sigma*arg)

        out = freq_term * gauss_term

        if self.time:
            time = input[..., 3:]
            lin = self.linear(time)
            omega = self.omega * lin
            sigma = self.sigma * lin
            time_term = torch.exp(1j*omega - sigma.square())
            out = out * time_term
        return out.real
    

class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers,
            hidden_inr,
            time,
            skip=True,
            omega=10.,
            sigma=10.,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalGaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(hidden_dim, time, omega, sigma))

        self.nonlin = hidden_inr_dict[hidden_inr]

        for i in range(hidden_layers):
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