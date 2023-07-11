import numpy as np
import torch
from src.models import initializers as init
from src.modules.sine import Sine
from torch import nn
from scipy.special import sph_harm

class SphericalHarmonicsEmbedding(nn.Module):
    def __init__(self, order):
        super(SphericalHarmonicsEmbedding, self).__init__()
        self.order = order
        # Combine Real Image
        self.coefficients = nn.Parameter(torch.randn(order, 2*order+1 ))# , device='cpu')
        
        # Reduce waste of param
        # self.coefficients = nn.Parameter(torch.randn(order*(order+2))) 

        # Split Real Image
        # self.coefficients_sin = nn.Parameter(torch.randn(order, 2*order+1 ))# , device='cpu')
        # self.coefficients_cos = nn.Parameter(torch.randn(order, 2*order+1 ))# , device='cpu')
        

    def forward(self, x):
        '''
            x : [batch, node_num, 2] # 2 = (theta, phi)
        '''
        list_y_stack_batch = []
        batch_size = x.shape[0]

        for i in range(batch_size):
            time_i = x[i,:,2]
            y = torch.zeros(x.shape[1], dtype=torch.cfloat).cuda() # torch.Size([5000]) # theta, phi를 하나의 scalar로 embedding 하는 거니까.
            list_y_stack_order = []    
            # index=0   

            for l in range(self.order):
                for m in range(-l, l+1):
                    theta = x[i,:,0].detach().cpu().numpy()
                    phi = x[i,:,1].detach().cpu().numpy()

                    # Combine Real, Image
                    #1# sph_harm_emb_real = torch.Tensor(sph_harm(m,l,theta,phi).real).cuda()
                    sph_harm_emb = torch.from_numpy(sph_harm(m,l,theta,phi)).cuda()
                    

                    # Split Real, Image
                    # sph_harm_emb_real = torch.Tensor(sph_harm(m,l,theta,phi).real).cuda()
                    # sph_harm_emb_imag = torch.Tensor(sph_harm(m,l,theta,phi).imag).cuda() # doesn't work due to dtype problem
                    

                    #-----------------------------------------------------------------------------------------------------#
                    # Only use real
                    y += self.coefficients[l,m+self.order] * sph_harm_emb# from_numpy(sph_harm(m,l,theta,phi).real)

                    # Reduce waste of coefficient
                    # y += self.coefficients[index] * sph_harm_emb# from_numpy(sph_harm(m,l,theta,phi).real)

                    # Split Real Image
                    # y += self.coefficients_sin[l,m+self.order] * sph_harm_emb_real# from_numpy(sph_harm(m,l,theta,phi).real)
                    # y += self.coefficients_cos[l,m+self.order] * sph_harm_emb_imag# from_numpy(sph_harm(m,l,theta,phi).imag)
                    
                    # y += sph_harm_emb
                list_y_stack_order.append(y) # Making (5000, order) tensor
            list_y_stack_order.append(time_i)
            tensor_y_stacked_order = torch.stack(list_y_stack_order, dim=1) # Made (5000, order+1) tensor (order + time)
            list_y_stack_batch.append(tensor_y_stacked_order) # Making (batch, 5000, order+1) tensor

        embedded_y = torch.stack(list_y_stack_batch, dim=0)
        return embedded_y

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
        self.spherical_harmonics_embedding = SphericalHarmonicsEmbedding(self.max_order)

        # Modules
        self.model = nn.ModuleList()
        in_dim = input_dim
        out_dim = hidden_dim
        # print('input dim : ',input_dim) # 35 or 4
        # print('out_dim : ',out_dim) # 512
        for i in range(n_layers):   

            layer = nn.Linear(in_dim, out_dim)

            # print('geometric_init is ',geometric_init) # default : False
            # print('all_sine is ',all_sine) # default : False
            # print('sine is ',sine) # default : False
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
        # print('input shape before sh embedding : ',x.shape)
        # x_in = self.spherical_harmonics_embedding(x) # spherical harmonics embedding
        x = self.spherical_harmonics_embedding(x)
        x_in=x
        # print('input1 shape after sh embedding : ',x.shape)
        # print('input2 shape after sh embedding : ',x_in.shape)

        for i, layer in enumerate(self.model):
            if i == self.skip_at:
                x = torch.cat([x, x_in], dim=-1)
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
