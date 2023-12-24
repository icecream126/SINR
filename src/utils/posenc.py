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
        # assert self.n_levels < 30 # healpy can deal with n_levels below than 30
        self.F = F
        
        # sizes = [12 * ((2 ** (level - 1)) ** 2 + 2) for level in range(1, n_levels + 1)] # [36, 72, 216, 792]
        # max_size = max(sizes)
        param_tensor = torch.randn(n_levels, self.n_pix, F)
        self.params = nn.Parameter(param_tensor)

        for i in range(n_levels):
            nn.init.uniform_(self.params[i], a=-0.0001, b=0.0001)
        
    
    def get_all_level_pixel_index(self, x, device):
        '''
        np_rad_x (torch.tensor) : [batch, 2]
        '''
        all_level_pixel_index = None
        for i in range(self.n_levels):
            nside = 2**i
            pixel_index = hp.ang2pix(nside = nside, theta = x[...,1].detach().cpu().numpy(), phi = x[...,0].detach().cpu().numpy(), lonlat=True)
            pixel_index = torch.tensor(pixel_index).unsqueeze(0).to(device)
            if all_level_pixel_index is None:
                all_level_pixel_index = pixel_index
            else:
                all_level_pixel_index = torch.cat((all_level_pixel_index, pixel_index), dim=0)
        return all_level_pixel_index # [n_levels, batch]
    
    def get_all_level_pixel_latlon(self, all_level_pixel_index, device):
        '''
        all_level_pixel_index (torch.tensor) : [n_levels, batch, 1]
        '''
        all_level_pixel_latlon = None
        all_level_pixel_index = all_level_pixel_index.detach().cpu().numpy() # [level, batch]
        for i in range(self.n_levels):
            nside = 2**i
            pixel_index = all_level_pixel_index[i]
            pixel_latlon = hp.pix2ang(nside = nside, ipix = pixel_index , lonlat=False)
            pixel_lat = torch.tensor(pixel_latlon[0]).unsqueeze(-1)
            pixel_lon = torch.tensor(pixel_latlon[1]).unsqueeze(-1)
            pixel_latlon = torch.cat((pixel_lat, pixel_lon), dim=-1).unsqueeze(0).to(device)
            if all_level_pixel_latlon is None:
                all_level_pixel_latlon = pixel_latlon
            else:
                all_level_pixel_latlon = torch.cat((all_level_pixel_latlon, pixel_latlon), dim=0)
        return all_level_pixel_latlon # [n_levels, batch, 2]
    
    def get_all_level_neigh_index(self, all_level_pixel_index, device):
        '''
        all_level_pixel_index (torch.tensor) : [n_levels, batch, 1]
        '''
        all_level_neigh_index = None
        for i in range(self.n_levels):
            nside = 2**i
            pixel_index = all_level_pixel_index[i]
            neigh_index = hp.get_all_neighbours(nside, pixel_index.detach().cpu().numpy())
            neigh_index = torch.tensor(neigh_index).to(device).unsqueeze(0)
            if all_level_neigh_index is None:
                all_level_neigh_index = neigh_index
            else:
                all_level_neigh_index = torch.cat((all_level_neigh_index, neigh_index), dim=0)
        return all_level_neigh_index # [n_levels, 8, batch, 1]
    
    
    def get_great_circle(self, x_rads, neighbor_rads):
        lat1 = x_rads[..., 0]
        lon1 = x_rads[..., 1]

        lat2 = neighbor_rads[..., 0]
        lon2 = neighbor_rads[..., 1]
        
        lat1[torch.isinf(lat1)] = lat1.min()
        lat2[torch.isinf(lat2)] = lat2.min()
        lon1[torch.isinf(lon1)] = lon1.min()
        lon2[torch.isinf(lon2)] = lon2.min()
        
        if self.great_circle:
        
        ###### Great circle distance ####
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
            
            # Ensure no negative values inside the square root
            sqrt_a = torch.sqrt(torch.clamp(a, min=0))
            
            dist = 2 * torch.atan2(sqrt_a, torch.sqrt(torch.clamp(1 - a, min=0)))
        ###################################       
        
        else:
        # Bilinear interpolation on Euclidean
            dlon = torch.pow(lon2 - lon1,2)
            dlat = torch.pow(lat2 - lat1,2)
            
            dist = torch.sqrt(dlon + dlat)
        dist[torch.isinf(dist)]=dist.min() # inf가 생기는 이유는 -1 neighbor를 max+1로 바꾸어놨기 때문. level 0에서 max+1의 lat, lon 값은 없을 것임. 

        
        return dist # [level, 8 * batch]
    
    def interpolate(self, all_level_my_reps, all_level_neigh_reps, all_level_my_latlon, all_level_neigh_latlon, all_level_neigh_mask):
        # Given center(my) pixel position,
        '''
            all_level_my_reps      : [n_levels, batch,   self.F]  ex) [4,  512, 2]
            all_level_my_latlon    : [n_levels, batch,        2]  ex) [4,  512, 2]
            all_level_neigh_reps   : [n_levels, 8*batch, self.F]  ex) [4, 4096, 2]
            all_level_neigh_latlon : [n_levels, 8*batch,      2]  ex) [4, 4096, 2]
        '''
        
        all_level_my_latlon = all_level_my_latlon.unsqueeze(1).repeat(1, 8, 1, 1)          # [level, neigh, batch, 2]
        all_level_my_latlon = all_level_my_latlon.reshape(all_level_neigh_latlon.shape)          # [level, neigh*batch , 2]
        # all_level_neigh_latlon = all_level_neigh_latlon.reshape(all_level_my_latlon.shape) # [level, neigh, batch, 2]
        
        distances = self.get_great_circle(all_level_my_latlon, all_level_neigh_latlon) # [level, neigh * batch]
        # weight = 1/distances
        weight = distances.max() - distances
        weight = weight.reshape(weight.shape[0], 8, -1) # [level, neigh, batch]
        weight = weight.unsqueeze(-1).repeat(1,1,1,self.F) # [level, neigh, batch, 2]
        weight = weight.reshape(weight.shape[0], -1, self.F) # [level, neigh*batch, 2]
        
        # [level, neigh*batch] => [level, neigh, batch, 2]
        all_level_neigh_mask = all_level_neigh_mask.reshape(all_level_neigh_mask.shape[0], 8, all_level_neigh_mask.shape[1]//8).unsqueeze(-1).repeat(1,1,1,self.F) 
        
        out_reps = torch.multiply(all_level_neigh_reps, weight)
        out_reps = out_reps.reshape(out_reps.shape[0], 8, -1, self.F) # [level, neigh, batch, self.F]
        
        out_reps = out_reps * all_level_neigh_mask
        
        out_reps = torch.sum(out_reps, dim=1) # [level, batch, self.F]
        # out_reps = torch.add(out_reps, all_level_my_reps) # [level, batch, self.F]
        
        out_reps = out_reps.reshape(self.n_levels, -1).t()
        out_reps = out_reps.reshape(-1, self.n_levels * self.F)
        return out_reps
    
    def index_preprocessing(self, index):
        mask = index == -1 
        index[mask] = torch.max(index)+1
        mask = ~mask
        return index, mask
    
    def get_all_level_rep(self, all_level_pixel_index, all_level_neigh_index, all_level_pixel_latlon, device):
        '''
        all_level_pixel_index  : [n_levels, batch]
        all_level_neigh_index  : [n_levels, 8, batch]
        all_level_pixel_latlon  : [n_levels, batch, 2]
        '''
        
        # [n_level, 8, batch] => [n_level, 8*batch]
        all_level_neigh_index = all_level_neigh_index.reshape(all_level_neigh_index.shape[0], all_level_neigh_index.shape[1]*all_level_neigh_index.shape[2]) # [level, 8*batch]
        

        all_level_neigh_index, all_level_neigh_mask = self.index_preprocessing(all_level_neigh_index) # [level, 8, batch], [level, 8, batch]
        all_level_pixel_index, _ = self.index_preprocessing(all_level_pixel_index) # [level, batch]
        
        all_level_neigh_reps = torch.gather(self.params, 1, all_level_neigh_index.unsqueeze(-1).expand(-1, -1, self.params.size(-1))) # [4, 4096, 2] [level, 8*batch, F]
        all_level_my_reps = torch.gather(self.params, 1, all_level_pixel_index.unsqueeze(-1).expand(-1, -1, self.params.size(-1))) # [4, 512, 2]   [level, batch, F]     
        all_level_neigh_latlon = self.get_all_level_pixel_latlon(all_level_neigh_index, device) # [n_levels, 8*batch, 2]        
        
        # Adding  hp.get_interp_val here
        # for i in range(self.n_levels):
        #     npix = hp.nside2npix(2**i)
        #     level_m = self.params[i][:npix,:] # [1, npix, self.F] # map of specific level with corresponding npix, self.F size
        #     # for j in range(self.F):
        #     #     level_feature_m = level_m[j] # [1, npix, 1] # map of speicific level, self.F with corresponding npix size
            
        #     hp.get_interp_val(level_m, )
                
        
        out = self.interpolate(all_level_my_reps, all_level_neigh_reps, all_level_pixel_latlon, all_level_neigh_latlon, all_level_neigh_mask)
        
        return out # [batch, n_levels * self.F]
    


    def visualize_pixel(self,x, all_level_pixel_latlon, all_level_pixel_index, all_level_neigh_index):
        # all_level_pixel_index  : [  4, 512     ]
        # all_level_neigh_index  : [  4,   8, 512]
        # x                      : [512,   2,    ]
        # all_level_pixel_latlon : [  4, 512,   2]

        
        for i in range(self.n_levels) :
            single_level_pixel_index = all_level_pixel_index[i] # [512,]
            single_level_neigh_index = all_level_neigh_index[i] # [8, 512]
            
            for j in range(all_level_pixel_index.shape[-1]):
                center_pixel = single_level_pixel_index[...,j] # [1,]
                neighbor_pixels = single_level_neigh_index[...,j] # [8,]
                
                nside = 2**i
                npix = hp.nside2npix(nside)
                healpix_map = np.zeros(npix)
                
                healpix_map[center_pixel.detach().cpu().numpy()] = 1
                healpix_map[neighbor_pixels.detach().cpu().numpy()] = 2
                
                lat = x[j][...,0]
                lon = x[j][...,1]
                
                cmap = plt.cm.viridis
                cmap.set_under('white') # bg color
                cmap.set_over('blue') # color for neighbors
                cmap.set_bad('red') # color for center
                hp.mollview(healpix_map, min=0.9, max=2.1, cmap=cmap, title=f"Healpix Pixels Visualization_{lat},{lon}")
                plt.savefig(f'level_{i}_pix_{j}.png')
                if j==3: break
            
            if i==3: break
                # plt.show()
                
                
        
    
    def forward(self, x):
        '''
        x : [batch, 2]
            x[...,0] : lat : [-90, 90]
            x[...,1] : lon : [0, 360)
        '''
        device = x.device
        
        all_level_pixel_index = self.get_all_level_pixel_index(x, device) # [n_levels, batch]
        all_level_neigh_index = self.get_all_level_neigh_index(all_level_pixel_index, device) # [n_levels, 8, batch] # 대부분은 맞긴한데... 맞는듯 틀리는듯..
        all_level_pixel_latlon = self.get_all_level_pixel_latlon(all_level_pixel_index, device) # [n_levels, batch, 2]

        all_level_rep = self.get_all_level_rep(all_level_pixel_index, all_level_neigh_index, all_level_pixel_latlon, device) # [n_levels, batch, self.F]

        return all_level_rep.float()
    

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
