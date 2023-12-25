# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L222
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import healpy as hp
from healpy.rotator import Rotator
import matplotlib.pyplot as plt


LAT_MIN = -90.0
LON_MIN = 0.0

LAT_MAX = 90.0
LON_MAX = 360.0


class HealEncoding(nn.Module):
    def __init__(self, n_levels, F, great_circle):
        super().__init__()
        self.great_circle = great_circle
        self.n_levels = n_levels
        self.n_side = 2**(n_levels-1)
        self.n_pix = hp.nside2npix(self.n_side)
        self.F = F
        
        param_tensor = torch.randn(n_levels, self.n_pix, F)
        self.params = nn.Parameter(param_tensor)

        for i in range(n_levels):
            nn.init.uniform_(self.params[i], a=-0.0001, b=0.0001)

    def forward(self, x):
        '''
        x : [batch, 2]
            x[...,0] : lat : [-90, 90]
            x[...,1] : lon : [0, 360)
        '''
        lat = x[...,0].detach().cpu().numpy()
        lon = x[...,1].detach().cpu().numpy()
        all_level_reps = []
        for i in range(self.n_levels):
            # center_pix = hp.ang2pix(nside=2**i, theta=lon, phi=lat, lonlat=True) # [batch]
            # center_pix = torch.tensor(center_pix, device=x.device)
            neigh_pix, neigh_weight = hp.get_interp_weights(nside=2**i, theta = lon, phi = lat, lonlat=True) #[4, batch], [4, batch]
            neigh_pix = torch.tensor(neigh_pix, device=x.device).flatten() # [4*batch]
            neigh_weight = torch.tensor(neigh_weight, device=x.device)

            # center_reps = torch.gather(self.params[i], 0, center_pix.unsqueeze(-1).expand(-1, self.F)) # [batch, 2] [2048, F]
            neigh_reps = torch.gather(self.params[i], 0, neigh_pix.unsqueeze(-1).expand(-1, self.F)) # [batch*4, 2] [2048, F]
            neigh_reps = neigh_reps.reshape(4, x.shape[0], self.F) # [batch, 4, 2]
            neigh_weight = neigh_weight.unsqueeze(-1).repeat(1,1,self.F) # [batch, 4] ->[batch, 4, 2]
            
            neigh_reps = torch.multiply(neigh_reps, neigh_weight) # [batch, 4, 2]
            neigh_reps = neigh_reps.sum(dim=0) # [batch, 2]
            # out_reps = torch.add(center_reps, neigh_reps) # [batch, 2]
            all_level_reps.append(neigh_reps)
        all_level_reps = torch.stack(all_level_reps, dim=-1) # [512, 2, 11]
        all_level_reps = all_level_reps.reshape(x.shape[0],-1) # [512, 22]

        return all_level_reps.float()
    

class NGP_INTERP_ENC(nn.Module):
    def __init__(
        self,
        geodesic_weight,
        F=2,
        L=16,
        T=14,
        input_dim=2,
        finest_resolution=512,
        base_resolution=16,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.geodesic_weight = geodesic_weight
        self.T = T
        self.L = L
        self.N_min = base_resolution
        self.N_max = finest_resolution
        self._F = F

        self.b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.L - 1))

        # Integer list of each level resolution: N1, N2, ..., N_max
        # shape [1, 1, L, 1]
        self._resolutions = nn.Parameter(
            torch.from_numpy(
                np.array(
                    [np.floor(self.N_min * (self.b**i)) for i in range(self.L)],
                    dtype=np.int64,
                )
            ).reshape(1, 1, -1, 1),
            False,
        )  #

        # Init hash tables for all levels
        # # Each hash table shape [2^T, F]
        # self.embeddings   = nn.ModuleList([
        #     nn.Embedding((self.lat_shape+1)*(self.lon_shape+1),
        #                             self.n_features_per_level) for i in range(n_levels)])
        self._hash_tables = nn.ModuleList(
            [nn.Embedding(int(2**self.T), int(self._F)) for _ in range(self.L)]
        )

        for i in range(self.L):
            nn.init.uniform_(
                self._hash_tables[i].weight, -1e-4, 1e-4
            )  # init uniform random weight

        """from Nvidia's Tiny Cuda NN implementation"""
        self._prime_numbers = nn.Parameter(
            torch.tensor(
                [
                    1,
                    2654435761,
                    805459861,
                    3674653429,
                    2097192037,
                    1434869437,
                    2165219737,
                ]
            ),
            requires_grad=False,
        )

        """
        a helper tensor which generates the voxel coordinates with shape [1,input_dim,1,2^input_dim]
        2D example: [[[[0, 1, 0, 1]],
                      [[0, 0, 1, 1]]]]
        3D example: [[[[0, 1, 0, 1, 0, 1, 0, 1]],  
                      [[0, 0, 1, 1, 0, 0, 1, 1]],  
                      [[0, 0, 0, 0, 1, 1, 1, 1]]]]
        For n-D, the i-th input add the i-th "row" here and there are 2^n possible permutations
        """
        border_adds = np.empty((self.input_dim, 2**self.input_dim), dtype=np.int64)

        for i in range(self.input_dim):
            pattern = np.array(
                ([0] * (2**i) + [1] * (2**i)) * (2 ** (self.input_dim - i - 1)),
                dtype=np.int64,
            )
            border_adds[i, :] = pattern
        self._voxel_border_adds = nn.Parameter(
            torch.from_numpy(border_adds).unsqueeze(0).unsqueeze(2), False
        )  # helper tensor of shape [1,input_dim,1,2^input_dim]

    def _fast_hash(self, x: torch.Tensor):
        """
        Implements the hash function proposed by NVIDIA.
        Args:
            x: A tensor of the shape [batch_size, input_dim, L, 2^input_dim].
            This tensor should contain the vertices of the hyper cuber
            for each level.
        Returns:
            A tensor of the shape [batch_size, L, 2^input_dim] containing the
            indices into the hash table for all vertices.
        """
        tmp = torch.zeros(
            (
                x.shape[0],
                self.L,
                2**self.input_dim,
            ),  # shape [batch_size,L,2^input_dim]
            dtype=torch.int64,
            device=x.device,
        )
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(
                x[:, i, :, :]
                * self._prime_numbers[i],  # shape [batch_size,L,2^input_dim]
                tmp,
            )
        return torch.remainder(tmp, 2**self.T)  # mod 2^T

    def normalize_2d_x(self, x):
        lat_min = -90
        lat_max = 90
        lon_min = 0
        lon_max = 360

        x[..., 0] = (x[..., 0] - lat_min) / (lat_max - lat_min)
        x[..., 1] = (x[..., 1] - lon_min) / (lon_max - lon_min)

        return x

    def normalize_3d_x(self, x):
        min_val = -1
        max_val = 1
        for i in range(3):
            x[..., i] = (x[..., i] - min_val) / (max_val - min_val)

        # x[...,0] = (x[...,0]-lat_min)/(lat_max - lat_min)
        # x[...,1] = (x[...,1]-lon_min)/(lon_max - lon_min)

        return x

    def forward(self, x):
        """
        forward pass, takes a set of input vectors and encodes them

        Args:
            x: A tensor of the shape [batch_size, input_dim] of all input vectors.

        Returns:
            A tensor of the shape [batch_size, L*F]
            containing the encoded input vectors.
        """

        # 1. Scale each input coordinate by each level's resolution
        """
        elementwise multiplication of [batch_size,input_dim,1,1] and [1,1,L,1]
        """
        # if input is {theta, phi} coordinate, normalize each into [0, 1]
        if self.input_dim == 2:
            x = self.normalize_2d_x(x)
        # else:
        # x = self.normalize_3d_x(x)

        scaled_coords = torch.mul(
            x.unsqueeze(-1).unsqueeze(-1), self._resolutions
        )  # shape [batch_size,input_dim,L,1]
        # N_min=16이고, b=2.0일 때
        # L=0에서 coord * 16 * 2.0 ** 0
        # L=1에서 coord * 16 * 2.0 ** 1
        # ...
        # 즉, L dim의 의미는, 각 level의 resolution 만큼 coordinate에 곱해진 값을 저장
        # compute the floor of all coordinates
        if scaled_coords.dtype in [torch.float32, torch.float64]:
            grid_coords = torch.floor(scaled_coords).type(
                torch.int64
            )  # shape [batch_size,input_dim,L,1]
        else:
            grid_coords = scaled_coords  # shape [batch_size,input_dim,L,1]

        """
        add all possible permutations to each vertex
        obtain all 2^input_dim neighbor vertices at this voxel for each level
        """
        #
        grid_coords = torch.add(
            grid_coords, self._voxel_border_adds
        )  # shape [batch_size,input_dim,L,2^input_dim]

        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(
            grid_coords
        )  # hashed shape [batch_size, L, 2^input_dim]

        # 3. Look up the hashed indices
        looked_up = torch.stack(
            [
                # use indexing for nn.Embedding (check pytorch doc)
                # shape [batch_size,2^n,F] before permute
                # shape [batch_size,F,2^n] after permute
                self._hash_tables[i](hashed_indices[:, i]).permute(0, 2, 1)
                for i in range(self.L)
            ],
            dim=2,
        )  # shape [batch_size,F,L,2^n]

        # 4. Interpolate features using multilinear interpolation
        # 2D example: for point (x,y) in unit square (0,0)->(1,1)
        # bilinear interpolation is (1-x)(1-y)*(0,0) + (1-x)y*(0,1) + x(1-y)*(1,0) + xy*(1,1)
        weights = 1.0 - torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype))
        )  # shape [batch_size,input_dim,L,2^input_dim]
        weights = torch.prod(
            weights, axis=1, keepdim=True
        )  # shape [batch_size,1,L,2^input_dim]

        # sum the weighted 2^n vertices to shape [batch_size,F,L]
        # swap axes to shape [batch_size,L,F]
        # final shape [batch_size,L*F]
        output = (
            torch.sum(torch.mul(weights, looked_up), axis=-1)
            .swapaxes(1, 2)
            .reshape(x.shape[0], -1)
        )
        return output


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
class myGaussEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, mapping_size=256, scale=10, seed=None):
        super(myGaussEncoding, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.in_features = in_features
        self.mapping_size = mapping_size
        self.scale = scale
        self.gauss = torch.randn(in_features, mapping_size) * scale
        # self.gauss =np.random.normal(0,1,size=(in_features, mapping_size)) * scale

        # self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, x):
        # print('x : ',x.shape)
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        # batches, N = x.shape
        # self.gauss = self.gauss.cuda()
        # x = x.cuda()

        # self.gauss.shape : [3, 256]
        # x.shape : [128, 3]
        # sin.shape : [128, 256]
        # cos.shape : [128, 256]
        # out.shape : [128, 512]

        self.gauss = self.gauss.to(x.device)
        sin = torch.sin(2 * torch.pi * x @ self.gauss)
        cos = torch.cos(2 * torch.pi * x @ self.gauss)

        out = torch.cat([sin, cos], dim=1)
        # print("out shape",out.shape)
        return out


class SwaveletEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(
        self,
        in_features,
        batch_size,
        mapping_size=512,
        tau_0=10,
        omega_0=10,
        sigma_0=0.1,
    ):
        super().__init__()
        in_features = 2
        self.in_features = in_features
        self.mapping_size = mapping_size
        # self.scale = scale
        # self.gauss =torch.randn(in_features, mapping_size) * scale
        # self.tau_0 = tau_0
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        # self.tau = torch.randn(batch_size, in_features) # make it learnable Parameter
        # self.omega = torch.randn(batch_size, in_features)
        in_features = 1
        self.tau = nn.Parameter(2 * torch.rand(mapping_size, in_features) - 1)

        # self.tau = nn.Parameter(torch.empty(batch_size))#, in_features))
        self.omega = nn.Parameter(
            torch.empty(in_features, mapping_size)
        )  # , in_features))
        # self.omega = nn.Linear(in_features, mapping_size)

        self.sigma = nn.Parameter(torch.empty(mapping_size))
        self.phi_0 = nn.Parameter(torch.empty(mapping_size))

        nn.init.normal_(self.tau)
        # nn.init.normal_(self.omega)
        nn.init.normal_(self.sigma)

    def gabor_function(self, x):
        self.tau = self.tau.to(x.device)
        self.omega = self.omega.to(x.device)
        self.sigma = self.sigma.to(x.device)
        theta = x[..., 0]
        phi = x[..., 1]

        import pdb

        pdb.set_trace()

        freq_term = torch.sin(
            self.omega_0
            * self.omega
            * torch.tan(theta / 2)
            * torch.cos(self.phi_0 - phi)
        )
        gauss_term = torch.exp(-(1 / 2) * torch.pow(torch.tan(theta / 2), 2))

        out = freq_term * gauss_term / (1 + torch.cos(theta) + 1e-6)

        return out

    def forward(self, x):
        out = self.gabor_function(x)
        # import pdb
        # pdb.set_trace()
        return out
