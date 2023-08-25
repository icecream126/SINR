import torch
import numpy as np
from torch import nn
from math import ceil

from .model import MODEL
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega.
    
        If is_first=True, omega is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            is_first, 
            omega,
            **kwargs,
        ):
        super().__init__()

        self.omega = omega
        self.is_first = is_first
        
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega, 
                                             np.sqrt(6 / self.input_dim) / self.omega)
        
    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))
    
class INR(MODEL):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers, 
            time,
            skip,
            omega,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        self.nonlin = SineLayer
            
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(input_dim, 
                                    hidden_dim, 
                                    is_first=True, 
                                    omega=omega))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers/2):
                self.net.append(self.nonlin(hidden_dim+input_dim,
                                            hidden_dim,
                                            is_first=False,
                                            omega=omega))
            else:
                self.net.append(self.nonlin(hidden_dim,
                                            hidden_dim,
                                            is_first=False,
                                            omega=omega))

        final_linear = nn.Linear(hidden_dim, output_dim) 

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