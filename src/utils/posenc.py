# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L222
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



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


class SphereLearnableEncoder(nn.Module):
    def __init__(self, lat_shape, lon_shape, level, resolution):
        super(SphereLearnableEncoder, self).__init__() 
        self.lat_shape = lat_shape
        self.lon_shape = lon_shape
        self.level = level
        self.resolution = resolution
        
        
        self.latent_grids = nn.ParameterList()
        
        
        for i in range(level):
            h_grid = int(math.ceil(lat_shape / (2 ** i)))
            w_grid = int(math.ceil(lon_shape / (2 ** i)))

            param = (
                nn.Parameter(
                    torch.empty((1, 1, h_grid, w_grid)), requires_grad=True
                )
            )
            nn.init.uniform_(param)
            self.latent_grids.append(param)
            
        self.north_pole_param = nn.Parameter(torch.empty(level), requires_grad=True)
        self.south_pole_param = nn.Parameter(torch.empty(level), requires_grad=True)
        nn.init.uniform_(self.north_pole_param)
        nn.init.uniform_(self.south_pole_param)
            
    def coord2index(self, x):
        lat = x[...,0:1] # [1, 10, 1]
        lon = x[...,1:2] # [1, 10, 1]
        
        lon_min = 0.0
        lat_max = 90.0
        lat_idx = ((lat_max - lat) / self.resolution).round().long()
        lon_idx = ((lon - lon_min) / self.resolution).round().long()
        
        lat_idx = torch.clamp(lat_idx, 0, self.lat_shape-1)
        lon_idx = torch.clamp(lon_idx, 0, self.lon_shape-1)
        return lat_idx, lon_idx
        
    def forward(self, x):
        lat_idx, lon_idx = self.coord2index(x)
        
        lat_idx = lat_idx.squeeze()
        lon_idx = lon_idx.squeeze()
        
        # north, south pole mask
        north_pole_mask = (lat_idx == self.lat_shape-1)
        south_pole_mask = (lat_idx == 0)
        
        # non pole mask
        non_pole_mask = (lat_idx > 0) & (lat_idx < self.lat_shape - 1)
        
        if len(x.shape)==3:
            latent_reps = torch.zeros(x.shape[1], self.level, device = x.device)
        elif len(x.shape)==2:
            latent_reps = torch.zeros(x.shape[0], self.level, device = x.device)
        else:
            raise Exception('invalid input shape (must be len(x.shape)==3 or len(x.shape)==2)')
            
        for i in range(self.level):
            upsampled_grid = F.interpolate(self.latent_grids[i], size=(self.lat_shape, self.lon_shape), mode='bilinear', align_corners=False)
            upsampled_grid = upsampled_grid.squeeze(0).squeeze(0)
            
            valid_lat_idx = lat_idx[non_pole_mask]
            valid_lon_idx = lon_idx[non_pole_mask]
            non_pole_mask = non_pole_mask.squeeze()
            
            latent_reps[non_pole_mask, i] = upsampled_grid[valid_lat_idx, valid_lon_idx]
        
        if not torch.all(north_pole_mask == False):
            latent_reps[north_pole_mask, :] = self.north_pole_param.expand_as(latent_reps[north_pole_mask, :])
        if not torch.all(south_pole_mask == False):
            latent_reps[south_pole_mask, :] = self.south_pole_param.expand_as(latent_reps[south_pole_mask, :])
        
        return latent_reps

class COOLCHIC_INTERP_ENC(nn.Module):
    def __init__(self, lat_shape, lon_shape, level, resolution):
        super(COOLCHIC_INTERP_ENC, self).__init__() 
        self.lat_shape = lat_shape
        self.lon_shape = lon_shape
        self.level = level
        self.resolution = resolution
        
        self.latent_grids = nn.ParameterList()
        for i in range(self.level):
            h_grid = int(math.ceil(lat_shape / (2 ** i)))
            w_grid = int(math.ceil(lon_shape / (2 ** i)))

            param = (
                nn.Parameter(
                    torch.empty((1, 1, h_grid, w_grid)), requires_grad=True
                )
            )
            nn.init.uniform_(param)
            self.latent_grids.append(param)

        
    def forward(self, x):
        # Assume that x has \theta, \phi information
        # Assume that x is [128, 2]
        
        # Upsample latent grids
        upsampled_grids = []
        for i in range(self.level):
            upsampled_grid = F.interpolate(self.latent_grids[i], size=(self.lat_shape, self.lon_shape), mode='bilinear', align_corners=False)
            upsampled_grids.append(upsampled_grid.squeeze(0).squeeze(0))
        tensor_upsampled_grids = torch.stack(upsampled_grids)

        # Convert coordinates to continuous grid indices
        lat = x[..., 0:1]
        lon = x[..., 1:2]
        lat_min, lon_min = -90.0, 0.0
        lat_max = 90.0
        lat_idx = ((lat_max - lat) / self.resolution)
        lon_idx = ((lon - lon_min) / self.resolution)

        # Perform bilinear interpolation
        latent_reps = []
        for level_grid in tensor_upsampled_grids:
            latent_rep = self.bilinear_interpolate(level_grid, lat_idx, lon_idx)
            latent_reps.append(latent_rep.squeeze())

        latent_reps = torch.stack(latent_reps, dim=-1)

        return latent_reps

    def bilinear_interpolate(self, grid, lat_idx, lon_idx):
        # Get the integer and fractional parts of the indices
        lat_floor = torch.floor(lat_idx).long()
        lon_floor = torch.floor(lon_idx).long()
        lat_ceil = lat_floor + 1
        lon_ceil = lon_floor + 1

        # Clip to range to avoid out of bounds error
        lat_floor = torch.clamp(lat_floor, 0, grid.shape[0] - 1)
        lon_floor = torch.clamp(lon_floor, 0, grid.shape[1] - 1)
        lat_ceil = torch.clamp(lat_ceil, 0, grid.shape[0] - 1)
        lon_ceil = torch.clamp(lon_ceil, 0, grid.shape[1] - 1)

        # Get the values at the corner points
        # point와 인접한 grid의 latent representation 가져오기
        # values_ff (floor lat, floor lon): lower-left corner
        # values_fc (floor lat, ceil  lon): lower-right corner
        # values_cf (ceil  lat, floor lon): upper-left corner
        # values_cc (ceil  lat, ceil  lon): upper-right corner
        
        values_ff = grid[lat_floor, lon_floor]
        values_fc = grid[lat_floor, lon_ceil]
        values_cf = grid[lat_ceil, lon_floor]
        values_cc = grid[lat_ceil, lon_ceil]

        # Calculate the weights for interpolation
        # 계산하고자 하는 point의 lat, lon index와 인접 latent grid와의 distance 구하기..
        lat_frac = (lat_idx - lat_floor)# .unsqueeze(-1)
        lon_frac = (lon_idx - lon_floor)# .unsqueeze(-1)

        # Perform the bilinear interpolation
        # values_f : interpolated value at the floor latitude (아래쪽 선)
        #            values_ff와 values_fc 사이에서의 interpolation (아래-왼쪽, 아래-오른쪽 지점 사이의 interpolation)
        # values_c : interpolated value at the ceil latitude  (위쪽  선)
        #            values_cf와 values_cc 사이에서의 interpolation (위쪽-왼쪽, 위쪽-오른쪽 지점 사이의 interpolation)
        # interpolated_values: 위쪽과 아래쪽 사이에서의 interpolation
        values_f = values_ff + lon_frac * (values_fc - values_ff)
        values_c = values_cf + lon_frac * (values_cc - values_cf)
        interpolated_values = values_f + lat_frac * (values_c - values_f)

        return interpolated_values
    

# class NGP_INTERP_ENC(nn.Module):
#     def __init__(self, lat_shape, lon_shape, level, resolution):
#         super(NGP_INTERP_ENC, self).__init__()
#         self.lat_shape = lat_shape
#         self.lon_shape = lon_shape
#         self.level = level
#         self.resolution = resolution

#         self.latent_grids = nn.ParameterList()
#         for i in range(self.level):
#             h_grid = int(math.ceil(lat_shape / (2 ** i)))
#             w_grid = int(math.ceil(lon_shape / (2 ** i)))

#             param = nn.Parameter(torch.empty((1, 1, h_grid, w_grid)), requires_grad=True)
#             nn.init.uniform_(param)
#             self.latent_grids.append(param)

#     def forward(self, x):
#         latent_reps = []
#         for i, latent_grid in enumerate(self.latent_grids):
#             adjusted_resolution = self.resolution * (2 ** i)
#             latent_rep = self.bilinear_interpolate(latent_grid.squeeze(0).squeeze(0), x, adjusted_resolution)
#             latent_reps.append(latent_rep)

#         latent_reps = torch.stack(latent_reps, dim=-1)
#         return latent_reps

#     def bilinear_interpolate(self, grid, x, resolution):
#         lat, lon = x[..., 0], x[..., 1]
#         lat_min, lon_min = -90.0, 0.0
#         lat_max = 90.0

#         # Convert coordinates to continuous grid indices
#         lat_idx = (lat_max - lat) / resolution
#         lon_idx = (lon - lon_min) / resolution

#         # Get the integer and fractional parts of the indices
#         lat_floor = torch.floor(lat_idx).long()
#         lon_floor = torch.floor(lon_idx).long()
#         lat_ceil = lat_floor + 1
#         lon_ceil = lon_floor + 1

#         # Clip to range to avoid out of bounds error
#         lat_floor = torch.clamp(lat_floor, 0, grid.shape[0] - 1)
#         lon_floor = torch.clamp(lon_floor, 0, grid.shape[1] - 1)
#         lat_ceil = torch.clamp(lat_ceil, 0, grid.shape[0] - 1)
#         lon_ceil = torch.clamp(lon_ceil, 0, grid.shape[1] - 1)

#         # Get the values at the corner points
#         values_ff = grid[lat_floor, lon_floor]
#         values_fc = grid[lat_floor, lon_ceil]
#         values_cf = grid[lat_ceil, lon_floor]
#         values_cc = grid[lat_ceil, lon_ceil]

#         # Calculate the weights for interpolation
#         lat_frac = (lat_idx - lat_floor)# .unsqueeze(-1)
#         lon_frac = (lon_idx - lon_floor)# .unsqueeze(-1)

#         # Perform the bilinear interpolation
#         values_f = values_ff + lon_frac * (values_fc - values_ff)
#         values_c = values_cf + lon_frac * (values_cc - values_cf)
#         interpolated_values = values_f + lat_frac * (values_c - values_f)

#         return interpolated_values
    
    
class NGP_INTERP_ENC(nn.Module):
    def __init__(self, lat_shape, lon_shape, level, resolution):
        super(NGP_INTERP_ENC, self).__init__()
        self.lat_shape = lat_shape
        self.lon_shape = lon_shape
        self.level = level
        self.resolution = resolution

        self.embeddings = nn.Parameter(torch.empty(self.level, self.lat_shape, self.lon_shape))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def forward(self, x):
        latent_reps = []
        for i, latent_grid in enumerate(self.embeddings):
            adjusted_resolution = self.resolution * (2 ** i)
            latent_rep = self.bilinear_interpolate(latent_grid, x, adjusted_resolution)
            latent_reps.append(latent_rep)

        latent_reps = torch.stack(latent_reps, dim=-1)
        return latent_reps

    def bilinear_interpolate(self, grid, x, resolution):
        
        grid = grid.to(x.device)
        
        lat, lon = x[..., 0], x[..., 1]
        lat_min, lon_min = -90.0, 0.0
        lat_max = 90.0

        # Convert coordinates to continuous grid indices
        lat_idx = (lat_max - lat) / resolution
        lon_idx = (lon - lon_min) / resolution

        # Get the integer and fractional parts of the indices
        lat_floor = torch.floor(lat_idx).long()
        lon_floor = torch.floor(lon_idx).long()
        lat_ceil = lat_floor + 1
        lon_ceil = lon_floor + 1

        # Clip to range to avoid out of bounds error
        lat_floor = torch.clamp(lat_floor, 0, grid.shape[0] - 1)
        lon_floor = torch.clamp(lon_floor, 0, grid.shape[1] - 1)
        lat_ceil = torch.clamp(lat_ceil, 0, grid.shape[0] - 1)
        lon_ceil = torch.clamp(lon_ceil, 0, grid.shape[1] - 1)

        # Get the values at the corner points
        values_ff = grid[lat_floor, lon_floor]
        values_fc = grid[lat_floor, lon_ceil]
        values_cf = grid[lat_ceil, lon_floor]
        values_cc = grid[lat_ceil, lon_ceil]

        # Calculate the weights for interpolation
        lat_frac = (lat_idx - lat_floor)# .unsqueeze(-1)
        lon_frac = (lon_idx - lon_floor)# .unsqueeze(-1)

        # Perform the bilinear interpolation
        values_f = values_ff + lon_frac * (values_fc - values_ff)
        values_c = values_cf + lon_frac * (values_cc - values_cf)
        interpolated_values = values_f + lat_frac * (values_c - values_f)

        return interpolated_values
    
    
    
    
class LearnableEncoding(nn.Module):
    def __init__(self, lat_shape, lon_shape, level, resolution):
        super(LearnableEncoding, self).__init__() 
        self.lat_shape = lat_shape
        self.lon_shape = lon_shape
        self.level = level
        self.resolution = resolution
        
        self.latent_grids = nn.ParameterList()
        for i in range(self.level):
            h_grid = int(math.ceil(lat_shape / (2 ** i)))
            w_grid = int(math.ceil(lon_shape / (2 ** i)))

            param = (
                nn.Parameter(
                    torch.empty((1, 1, h_grid, w_grid)), requires_grad=True
                )
            )
            nn.init.uniform_(param)
            self.latent_grids.append(param)
            
    def coord2index(self, x):
        
        lat = x[...,0:1]
        lon = x[...,1:2]
        
        # lat_min = -90.0
        lon_min = 0.0
        lat_max = 90.0
        # lon_max 
        lat_idx = ((lat_max - lat) / self.resolution).round().long()
        lon_idx = ((lon - lon_min) / self.resolution).round().long()
        
        idx = torch.cat((lat_idx, lon_idx), dim=-1).squeeze(0)
    
        return idx
        
    def forward(self, x):
        
        # Assume that x has \theta, \phi information
        # Assume that x is [128, 2]
        
        # self.latent_grids에 저장되어있는 각 latent_grids를 upsampling 시킨 후,
        # x에 해당하는 latent_grids들을 빼온다
        
        # Upsample latent grids
        # upsampled_grids = torch.zeros((self.level, self.lat_shape, self.lon_shape))
        upsampled_grids = []
        for i in range(self.level):
            upsampled_grid = F.interpolate(self.latent_grids[i], size=(self.lat_shape, self.lon_shape), mode='bilinear', align_corners=False)
            upsampled_grids.append(upsampled_grid.squeeze(0).squeeze(0))
        # x의 위치에 해당하는 grid 정보들을 빼와야함.
        tensor_upsampled_grids = torch.stack(upsampled_grids)
        
        idx = self.coord2index(x)
        # latent_reps = upsampled_grids[idx]
        lat_idx = idx[:,0]# .unsqueeze(1).expand(-1, tensor_upsampled_grids.size(0))
        lon_idx = idx[:,1]# .unsqueeze(1).expand(-1, tensor_upsampled_grids.size(0))
        
        latent_reps = tensor_upsampled_grids[:, lat_idx, lon_idx]
        latent_reps = latent_reps.T
        
        return latent_reps

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
        
        # self.gauss.shape : [3, 256]
        # x.shape : [128, 3]
        # sin.shape : [128, 256]
        # cos.shape : [128, 256]
        # out.shape : [128, 512]
        
        self.gauss = self.gauss.to(x.device)
        sin = torch.sin(2 * torch.pi * x @ self.gauss)
        cos = torch.cos(2 * torch.pi * x @ self.gauss)

        out = torch.cat([sin, cos], dim=1)
        return out





class EwaveletEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, batch_size, mapping_size=512, tau_0=10, omega_0=10, sigma_0 = 0.1):
        super().__init__()

        self.in_features = in_features
        self.mapping_size = mapping_size
        # self.scale = scale
        # self.gauss =torch.randn(in_features, mapping_size) * scale
        # self.tau_0 = tau_0
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        # self.tau = torch.randn(batch_size, in_features) # make it learnable Parameter
        # self.omega = torch.randn(batch_size, in_features)
        in_features=1
        self.tau = nn.Parameter(2*torch.rand(mapping_size, in_features)-1)
        
        # self.tau = nn.Parameter(torch.empty(batch_size))#, in_features))
        # self.omega = nn.Parameter(torch.empty(in_features, mapping_size))#, in_features))
        self.omega = nn.Linear(in_features, mapping_size)
        self.sigma = nn.Parameter(torch.empty(mapping_size))
        
        
        nn.init.normal_(self.tau)
        # nn.init.normal_(self.omega)
        nn.init.normal_(self.sigma)
        

    def gabor_function(self, x):
        "e^{ -pi (t-\tau)^2 e^{=j\omega t} }"
        "g_nm(k) = s(k-m*\tau_0) * e^{j\omega * n * k}"
        "Simplify to the following code : "
        self.tau = self.tau.to(x.device)
        self.omega = self.omega.to(x.device)
        self.sigma = self.sigma.to(x.device)
        
        for i in range(self.in_features):
            coord = x[...,i].unsqueeze(-1)
        
            A = (coord ** 2).sum(-1)[..., None]
            B = (self.tau ** 2).sum(-1)[None, :]
            C = 2 * coord @ self.tau.T 
            
            D = (A + B - C)
            gauss_term = torch.exp(-0.5 * D * self.sigma[None, :])
            freq_term = torch.sin(self.omega(coord))
            
            if i==0:
                out = gauss_term * freq_term
            else:
                out = torch.cat([out, gauss_term * freq_term], dim=-1)
            
        return out
    
    def forward(self, x):        
        out =  self.gabor_function(x)
        # import pdb
        # pdb.set_trace()
        return out





class SwaveletEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, batch_size, mapping_size=512, tau_0=10, omega_0=10, sigma_0 = 0.1):
        super().__init__()
        in_features=2
        self.in_features = in_features
        self.mapping_size = mapping_size
        # self.scale = scale
        # self.gauss =torch.randn(in_features, mapping_size) * scale
        # self.tau_0 = tau_0
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        # self.tau = torch.randn(batch_size, in_features) # make it learnable Parameter
        # self.omega = torch.randn(batch_size, in_features)
        in_features=1
        self.tau = nn.Parameter(2*torch.rand(mapping_size, in_features)-1)
        
        # self.tau = nn.Parameter(torch.empty(batch_size))#, in_features))
        self.omega = nn.Parameter(torch.empty(in_features, mapping_size))#, in_features))
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
        theta = x[...,0]
        phi = x[...,1]
        
        import pdb
        pdb.set_trace()
        
        
        freq_term = torch.sin(self.omega_0*self.omega*torch.tan(theta/2)*torch.cos(self.phi_0 - phi))
        gauss_term = torch.exp(-(1/2) * torch.pow(torch.tan(theta/2), 2))
        
        out = freq_term * gauss_term / (1+torch.cos(theta)+1e-6)
            
        return out
    
    def forward(self, x):        
        out =  self.gabor_function(x)
        # import pdb
        # pdb.set_trace()
        return out