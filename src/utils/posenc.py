# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L222
import math
import torch
from torch import nn
import numpy as np


class PosEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, num_frequencies=10):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1) 
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


# https://colab.research.google.com/github/ndahlquist/pytorch-fourier-feature-networks/blob/master/demo.ipynb#scrollTo=0ldSE8wbRyes
# We set mapping size 512 as a fixed hidden size
# So the only hyperparameter will be scale 
# We will set scale hyperparameter range as (2,4,6,8,10) (followed by GFF)
class GaussEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, mapping_size=256, scale=10):
        super().__init__()

        self.in_features = in_features
        self.mapping_size = mapping_size
        self.scale = scale
        self.gauss =torch.randn(in_features, mapping_size) * scale
        # self.gauss =np.random.normal(0,1,size=(in_features, mapping_size)) * scale

        # self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, x):
        # print('x : ',x.shape)
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        # batches, N = x.shape
        # self.gauss = self.gauss.cuda()
        # x = x.cuda()
        self.gauss = self.gauss.to(x.device)
        sin = torch.sin(2 * torch.pi * x @ self.gauss)
        cos = torch.cos(2 * torch.pi * x @ self.gauss)

        out = torch.cat([sin, cos], dim=1)
        return out

