import torch
from torch import nn
from math import ceil

class GaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            input_dim: Input features
            output_dim; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            sigma: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            is_first, 
            omega,
            sigma,
            **kwargs,
        ):
        super().__init__()

        self.omega = omega
        self.sigma = sigma
        self.input_dim = input_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega * lin
        sigma = self.sigma * lin
        out = torch.exp(1j*omega - sigma.abs().square())
        return out.real
    
class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim,
            hidden_layers, 
            skip=True,
            omega=10.,
            sigma=10.,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.nonlin = GaborLayer
            
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(input_dim,
                                    hidden_dim, 
                                    is_first=True,
                                    omega=omega,
                                    sigma=sigma))

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
        
        self.net.append(final_linear)
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers/2)+1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x