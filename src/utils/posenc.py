# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L222
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
import healpy as hp
# from astropy_healpix import HEALPix

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


LAT_MIN = -90.0
LON_MIN = 0.0

LAT_MAX = 90.0
LON_MAX = 360.0


class HealEncoding(nn.Module):
    def __init__(self, n_levels, F):
        super().__init__()
        # Healpix의 level에 따른 parameter grid 초기화
        self.n_levels = n_levels
        self.F = F
        self.params = nn.ModuleList([
            nn.Embedding(12*((2**(n_levels-1))**2+2), int(F)) for _ in range(self.n_levels)
        ]) # initialize parameters in 2d rectangular form, with the finest resolution * feature_dim for every level
        # self.hp_list = [HEALPix(nside=2**i, order='ring') for i in range (self.n_levels)]
        # number of corner = (12 * n_side)**2 +2
        # n_side = 2**level
    
    def get_great_circle(self, x_rads, neighbor_rads):
        
        # R=1.0
        lat1 = x_rads[...,0]
        lon1 = x_rads[...,1]
        
        lat2 = neighbor_rads[...,0]
        lon2 = neighbor_rads[...,1]
        
        dlon = lon2-lon1
        dlat = lat2-lat1
        
        # a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        # c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
        
        
        a = np.power(np.sin(dlat / 2),2) + np.multiply(np.multiply(np.cos(lat1), np.cos(lat2)) ,np.power(np.sin(dlon / 2),2))
        dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist = np.array(dist)

        # distance = R * c
        
        return dist
        
        
    def get_interp_rep(self, nside, x, neighbor_coords, neighbor_reps, my_reps):
        # interp_rep = self.interpolation(neighbor_coords, )
        # a_dist, b_dist, c_dist, d_dist = self.get_corner_dist(neighbor_coords,x) # ?
        # dist_sum = a_dist + b_dist + c_dist + d_dist
        # out = a_dist/dist_sum * a + 1/b_dist * b + 1/c_dist * c + 1/d_dist * d
        # print(neighbor_coords.shape)
        # x.shape : [2]
        # neighbor_reps.shape : [6,2]
        # adjust neighbor_coords into the shape
        '''Get interpolation feature of single point x'''
        np_x = x# .detach().cpu().numpy()
        x_rads = x
        neighbor_lat = torch.tensor(neighbor_coords[0]).unsqueeze(-1) # (6) -> (6,1)
        neighbor_lon = torch.tensor(neighbor_coords[1]).unsqueeze(-1) # (6) -> (6,1)
        
        neighbor_rads = torch.concat([neighbor_lat, neighbor_lon], dim=-1)
        
        # neighbor_vecs = hp.ang2vec(theta = neighbor_coords[...,0], phi = neighbor_coords[...,1])
        # x_rads = hp.ang2vec(theta = np_x[...,0], phi = np_x[...,1])
        x_rads = np.tile(x_rads, (neighbor_rads.shape[0],1))
        
        # distances = hp.rotator.angdist(neighbor_coords, x_rads)
        distances = self.get_great_circle(x_rads, neighbor_rads)
        sum_distances = np.sum(distances) + np.array(0.01)
        weight = distances / sum_distances
        
        neighbor_reps = neighbor_reps.detach().cpu().numpy()
        my_reps = my_reps.detach().cpu().numpy()
        
        out_reps = []
        for i in range(neighbor_reps.shape[1]):
            weight_mult = np.multiply(weight, neighbor_reps[...,i])
            sum_mult = np.add(np.sum(weight_mult),my_reps[...,i])
            out_reps.append(sum_mult)
        
        out_reps = np.array(out_reps)
        return out_reps     

        
    def interpolation(self,level, nside,x, np_rad_x, neighbor_index, my_index):
        # print('interpolation of heal encoding')
        # all_point_interp_reps = []
        
        # for i in range(self.n_levels):
        #     nside = 2**i
        #     neighbor_index = neighbor_indices[i]
        #     interp_reps=[]
        
        '''
        # x.shape : [512, 2]
        # np_rad_x.shape : [512, 2]
        # neighbor_index : [8, 512]
        '''
        
        # interp_reps = []
        all_point_interp_reps = torch.zeros(np_rad_x.shape[0], self.F)
        # my_index = hp.pix2nag(nside=nside, ipix = )
        for point in range(neighbor_index.shape[1]):
            point_index = my_index[...,point]
            point_neighbor_index = neighbor_index[...,point]
            point_neighbor_index = point_neighbor_index[point_neighbor_index!=-1] # discard invalid neighbors
            point_neighbor_coords = hp.pix2ang(nside = nside, ipix = point_neighbor_index)
            
            point_reps = self.params[level](torch.tensor(point_index).to(x.device))
            point_neighbor_reps = self.params[level](torch.tensor(point_neighbor_index).to(x.device))
            
            
            interp_rep = self.get_interp_rep(nside=nside, x=np_rad_x[point], neighbor_coords=point_neighbor_coords, neighbor_reps=point_neighbor_reps, my_reps = point_reps)
            # interp_reps.append(interp_rep)
            # interp_reps.append(interp_rep)
            all_point_interp_reps[point] = torch.tensor(interp_rep)
        # all_point_interp_reps.append(torch.tensor(interp_reps))
        # out = torch.tensor(all_point_interp_reps) 
        out = all_point_interp_reps
        return out

        
    def forward(self, x):
        # print('forward of heal encoding')
        
        # Input : [batch, 2]의 lon, lat coordinate
        # Output : 해당하는 input의 latent representation
        # 일단 grid의 coordinate 말고, center의 representation으로 진행하기
        
        # 0. Healpix의 level에 따른 parameter grid 초기화
        # 1. lon lat 이 주어졌을 때의 level별로 해당하는 parameter grid의 pixel index 알기
        # 2. Level 별로 parameter grid의 interpolation 수행
        # 3. Interpolation 된 grid를 concat해서 output으로 반환
        
        ### Pseudo Code###
        np_x = x.detach().cpu().numpy()
        
        rad_x = torch.deg2rad(x)
        rad_x[...,0] = torch.pi/2-rad_x[...,0] # adjust the range of latitude
        np_rad_x = rad_x.detach().cpu().numpy()
        
        indices = []
        # neighbor_indices = []
        all_level_rep = torch.zeros(self.n_levels, x.shape[0],self.F)
        all_level_rep = None
        for i in range(self.n_levels):
            nside = 2**i
            index = hp.ang2pix(nside=nside, theta = np_rad_x[...,0], phi = np_rad_x[...,1],lonlat=False)
            neighbor_index = hp.get_all_neighbours(nside=2**i, theta = np_rad_x[...,0], phi = np_rad_x[...,1],lonlat=False) # (8, 512)
            # index = hp.ang2pix(nside=2**i, theta = np.array(90) - np_x[...,0], phi = np_x[...,1],lonlat=True)
            # neighbor_index = hp.get_all_neighbours(nside=2**i, theta = np.pi/2 - np_x[...,0], phi = np_x[...,1],lonlat=False)
            # neighbor_index = neighbor_index[neighbor_index!=-1]
            # neighbor_indices.append(neighbor_index)
            
            # x.shape : [512, 2]
            # np_rad_x.shape : [512, 2]
            # neighbor_index : [8, 512]
            level_rep = self.interpolation(i, nside, x, np_rad_x, neighbor_index, index)
            # all_level_rep[i] = level_rep
            if all_level_rep is None:
                all_level_rep = level_rep
            else:
                all_level_rep = torch.cat((all_level_rep, level_rep), dim=-1)       
            
            
            # indices.append(index)
        # out = self.interpolation(rad_x, neighbor_indices)
        # out = torch.tensor(all_level_rep)
        out = all_level_rep
        out = out.to(x.device)
        return out
        
        
        
        

class SPHERE_NGP_INTERP_ENC(nn.Module):
    def __init__(self, geodesic_weight, F=2, L=16, T = 14, input_dim=2,  finest_resolution=512, base_resolution=16):        
        super().__init__()
        
        self.input_dim = input_dim
        self.geodesic_weight = geodesic_weight
        self.T = T
        self.L = L
        self.N_min = base_resolution
        self.N_max = finest_resolution
        self._F = F
        
        self.b = np.exp((np.log(self.N_max)-np.log(self.N_min))/(self.L-1))
        
        # Integer list of each level resolution: N1, N2, ..., N_max
        # shape [1, 1, L, 1]
        self._resolutions = nn.Parameter(
            torch.from_numpy(
                np.array([
                    np.floor(self.N_min * (self.b**i)) for i in range(self.L)
        ], dtype=np.int64)).reshape(1,1,-1,1),False) # 
        
        # Init hash tables for all levels
        # # Each hash table shape [2^T, F]
        # self.embeddings   = nn.ModuleList([
        #     nn.Embedding((self.lat_shape+1)*(self.lon_shape+1),
        #                             self.n_features_per_level) for i in range(n_levels)])
        self._hash_tables = nn.ModuleList([
            nn.Embedding(int(2**self.T),
                         int(self._F)) for _ in range(self.L)])
        
        for i in range(self.L):
            nn.init.uniform_(self._hash_tables[i].weight,-1e-4,1e-4) # init uniform random weight
            
        '''from Nvidia's Tiny Cuda NN implementation'''
        self._prime_numbers = nn.Parameter(
            torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]), requires_grad=False)
        
        '''
        a helper tensor which generates the voxel coordinates with shape [1,input_dim,1,2^input_dim]
        2D example: [[[[0, 1, 0, 1]],
                      [[0, 0, 1, 1]]]]
        3D example: [[[[0, 1, 0, 1, 0, 1, 0, 1]],  
                      [[0, 0, 1, 1, 0, 0, 1, 1]],  
                      [[0, 0, 0, 0, 1, 1, 1, 1]]]]
        For n-D, the i-th input add the i-th "row" here and there are 2^n possible permutations
        '''
        border_adds = np.empty((self.input_dim, 2**self.input_dim), dtype=np.int64)

        for i in range(self.input_dim):
            pattern = np.array(
                ([0] * (2**i) + [1] * (2**i)) * (2**(self.input_dim-i-1)),
                dtype=np.int64)
            border_adds[i, :] = pattern
        self._voxel_border_adds = nn.Parameter(
            torch.from_numpy(border_adds).unsqueeze(0).unsqueeze(2), False) # helper tensor of shape [1,input_dim,1,2^input_dim]
        
        

    def _fast_hash(self, x: torch.Tensor):
        '''
        Implements the hash function proposed by NVIDIA.
        Args:
            x: A tensor of the shape [batch_size, input_dim, L, 2^input_dim].
            This tensor should contain the vertices of the hyper cuber
            for each level.
        Returns:
            A tensor of the shape [batch_size, L, 2^input_dim] containing the
            indices into the hash table for all vertices.
        '''
        tmp = torch.zeros((x.shape[0], self.L, 2**self.input_dim), # shape [batch_size,L,2^input_dim]
                        dtype=torch.int64,
                        device=x.device)
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, i, :, :] * self._prime_numbers[i], # shape [batch_size,L,2^input_dim]
                                    tmp)
        return torch.remainder(tmp, 2**self.T) # mod 2^T
    
    def normalize_2d_x(self, x):
        lat_min = -90
        lat_max = 90
        lon_min = 0
        lon_max = 360
        
        x[...,0] = (x[...,0]-lat_min)/(lat_max - lat_min)
        x[...,1] = (x[...,1]-lon_min)/(lon_max - lon_min)
        
        return x
    def normalize_3d_x(self, x):
        min_val = -1
        max_val = 1
        for i in range(3):
            x[...,i] = (x[...,i]-min_val) / (max_val - min_val)
        
        # x[...,0] = (x[...,0]-lat_min)/(lat_max - lat_min)
        # x[...,1] = (x[...,1]-lon_min)/(lon_max - lon_min)
        
        return x
        
        

    
    def forward(self, x):
        '''
        forward pass, takes a set of input vectors and encodes them
        
        Args:
            x: A tensor of the shape [batch_size, input_dim] of all input vectors.

        Returns:
            A tensor of the shape [batch_size, L*F]
            containing the encoded input vectors.
        '''

        # 1. Scale each input coordinate by each level's resolution
        '''
        elementwise multiplication of [batch_size,input_dim,1,1] and [1,1,L,1]
        '''
        # if input is {theta, phi} coordinate, normalize each into [0, 1]
        if self.input_dim == 2:
            x = self.normalize_2d_x(x)
        # else:
            # x = self.normalize_3d_x(x)
            
        scaled_coords = torch.mul(x.unsqueeze(-1).unsqueeze(-1), self._resolutions) # shape [batch_size,input_dim,L,1]
        # N_min=16이고, b=2.0일 때
        # L=0에서 coord * 16 * 2.0 ** 0
        # L=1에서 coord * 16 * 2.0 ** 1
        # ...
        # 즉, L dim의 의미는, 각 level의 resolution 만큼 coordinate에 곱해진 값을 저장
        # compute the floor of all coordinates
        if scaled_coords.dtype in [torch.float32, torch.float64]:
            grid_coords = torch.floor(scaled_coords).type(torch.int64) # shape [batch_size,input_dim,L,1]
        else:
            grid_coords = scaled_coords # shape [batch_size,input_dim,L,1]
            
        '''
        add all possible permutations to each vertex
        obtain all 2^input_dim neighbor vertices at this voxel for each level
        '''
        # 
        grid_coords = torch.add(grid_coords, self._voxel_border_adds) # shape [batch_size,input_dim,L,2^input_dim] 

        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(grid_coords) # hashed shape [batch_size, L, 2^input_dim]

        # 3. Look up the hashed indices
        looked_up = torch.stack([
            # use indexing for nn.Embedding (check pytorch doc)
            # shape [batch_size,2^n,F] before permute
            # shape [batch_size,F,2^n] after permute
            self._hash_tables[i](hashed_indices[:,i]).permute(0, 2, 1)
            for i in range(self.L)
        ],dim=2) # shape [batch_size,F,L,2^n]

        # 4. Interpolate features using multilinear interpolation
        # 2D example: for point (x,y) in unit square (0,0)->(1,1)
        # bilinear interpolation is (1-x)(1-y)*(0,0) + (1-x)y*(0,1) + x(1-y)*(1,0) + xy*(1,1)
        weights = 1.0 - torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype))) # shape [batch_size,input_dim,L,2^input_dim]
        weights = torch.prod(weights, axis=1, keepdim=True) # shape [batch_size,1,L,2^input_dim]

        # sum the weighted 2^n vertices to shape [batch_size,F,L]
        # swap axes to shape [batch_size,L,F]
        # final shape [batch_size,L*F]
        output = torch.sum(torch.mul(weights, looked_up),
                         axis=-1).swapaxes(1, 2).reshape(x.shape[0], -1)
        return output




class NGP_INTERP_ENC(nn.Module):
    def __init__(self, geodesic_weight, F=2, L=16, T = 14, input_dim=2,  finest_resolution=512, base_resolution=16):        
        super().__init__()
        
        self.input_dim = input_dim
        self.geodesic_weight = geodesic_weight
        self.T = T
        self.L = L
        self.N_min = base_resolution
        self.N_max = finest_resolution
        self._F = F
        
        self.b = np.exp((np.log(self.N_max)-np.log(self.N_min))/(self.L-1))
        
        # Integer list of each level resolution: N1, N2, ..., N_max
        # shape [1, 1, L, 1]
        self._resolutions = nn.Parameter(
            torch.from_numpy(
                np.array([
                    np.floor(self.N_min * (self.b**i)) for i in range(self.L)
        ], dtype=np.int64)).reshape(1,1,-1,1),False) # 
        
        # Init hash tables for all levels
        # # Each hash table shape [2^T, F]
        # self.embeddings   = nn.ModuleList([
        #     nn.Embedding((self.lat_shape+1)*(self.lon_shape+1),
        #                             self.n_features_per_level) for i in range(n_levels)])
        self._hash_tables = nn.ModuleList([
            nn.Embedding(int(2**self.T),
                         int(self._F)) for _ in range(self.L)])
        
        for i in range(self.L):
            nn.init.uniform_(self._hash_tables[i].weight,-1e-4,1e-4) # init uniform random weight
            
        '''from Nvidia's Tiny Cuda NN implementation'''
        self._prime_numbers = nn.Parameter(
            torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]), requires_grad=False)
        
        '''
        a helper tensor which generates the voxel coordinates with shape [1,input_dim,1,2^input_dim]
        2D example: [[[[0, 1, 0, 1]],
                      [[0, 0, 1, 1]]]]
        3D example: [[[[0, 1, 0, 1, 0, 1, 0, 1]],  
                      [[0, 0, 1, 1, 0, 0, 1, 1]],  
                      [[0, 0, 0, 0, 1, 1, 1, 1]]]]
        For n-D, the i-th input add the i-th "row" here and there are 2^n possible permutations
        '''
        border_adds = np.empty((self.input_dim, 2**self.input_dim), dtype=np.int64)

        for i in range(self.input_dim):
            pattern = np.array(
                ([0] * (2**i) + [1] * (2**i)) * (2**(self.input_dim-i-1)),
                dtype=np.int64)
            border_adds[i, :] = pattern
        self._voxel_border_adds = nn.Parameter(
            torch.from_numpy(border_adds).unsqueeze(0).unsqueeze(2), False) # helper tensor of shape [1,input_dim,1,2^input_dim]
        
        

    def _fast_hash(self, x: torch.Tensor):
        '''
        Implements the hash function proposed by NVIDIA.
        Args:
            x: A tensor of the shape [batch_size, input_dim, L, 2^input_dim].
            This tensor should contain the vertices of the hyper cuber
            for each level.
        Returns:
            A tensor of the shape [batch_size, L, 2^input_dim] containing the
            indices into the hash table for all vertices.
        '''
        tmp = torch.zeros((x.shape[0], self.L, 2**self.input_dim), # shape [batch_size,L,2^input_dim]
                        dtype=torch.int64,
                        device=x.device)
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, i, :, :] * self._prime_numbers[i], # shape [batch_size,L,2^input_dim]
                                    tmp)
        return torch.remainder(tmp, 2**self.T) # mod 2^T
    
    def normalize_2d_x(self, x):
        lat_min = -90
        lat_max = 90
        lon_min = 0
        lon_max = 360
        
        x[...,0] = (x[...,0]-lat_min)/(lat_max - lat_min)
        x[...,1] = (x[...,1]-lon_min)/(lon_max - lon_min)
        
        return x
    def normalize_3d_x(self, x):
        min_val = -1
        max_val = 1
        for i in range(3):
            x[...,i] = (x[...,i]-min_val) / (max_val - min_val)
        
        # x[...,0] = (x[...,0]-lat_min)/(lat_max - lat_min)
        # x[...,1] = (x[...,1]-lon_min)/(lon_max - lon_min)
        
        return x
        
        

    
    def forward(self, x):
        '''
        forward pass, takes a set of input vectors and encodes them
        
        Args:
            x: A tensor of the shape [batch_size, input_dim] of all input vectors.

        Returns:
            A tensor of the shape [batch_size, L*F]
            containing the encoded input vectors.
        '''

        # 1. Scale each input coordinate by each level's resolution
        '''
        elementwise multiplication of [batch_size,input_dim,1,1] and [1,1,L,1]
        '''
        # if input is {theta, phi} coordinate, normalize each into [0, 1]
        if self.input_dim == 2:
            x = self.normalize_2d_x(x)
        # else:
            # x = self.normalize_3d_x(x)
            
        scaled_coords = torch.mul(x.unsqueeze(-1).unsqueeze(-1), self._resolutions) # shape [batch_size,input_dim,L,1]
        # N_min=16이고, b=2.0일 때
        # L=0에서 coord * 16 * 2.0 ** 0
        # L=1에서 coord * 16 * 2.0 ** 1
        # ...
        # 즉, L dim의 의미는, 각 level의 resolution 만큼 coordinate에 곱해진 값을 저장
        # compute the floor of all coordinates
        if scaled_coords.dtype in [torch.float32, torch.float64]:
            grid_coords = torch.floor(scaled_coords).type(torch.int64) # shape [batch_size,input_dim,L,1]
        else:
            grid_coords = scaled_coords # shape [batch_size,input_dim,L,1]
            
        '''
        add all possible permutations to each vertex
        obtain all 2^input_dim neighbor vertices at this voxel for each level
        '''
        # 
        grid_coords = torch.add(grid_coords, self._voxel_border_adds) # shape [batch_size,input_dim,L,2^input_dim] 

        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(grid_coords) # hashed shape [batch_size, L, 2^input_dim]

        # 3. Look up the hashed indices
        looked_up = torch.stack([
            # use indexing for nn.Embedding (check pytorch doc)
            # shape [batch_size,2^n,F] before permute
            # shape [batch_size,F,2^n] after permute
            self._hash_tables[i](hashed_indices[:,i]).permute(0, 2, 1)
            for i in range(self.L)
        ],dim=2) # shape [batch_size,F,L,2^n]

        # 4. Interpolate features using multilinear interpolation
        # 2D example: for point (x,y) in unit square (0,0)->(1,1)
        # bilinear interpolation is (1-x)(1-y)*(0,0) + (1-x)y*(0,1) + x(1-y)*(1,0) + xy*(1,1)
        weights = 1.0 - torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype))) # shape [batch_size,input_dim,L,2^input_dim]
        weights = torch.prod(weights, axis=1, keepdim=True) # shape [batch_size,1,L,2^input_dim]

        # sum the weighted 2^n vertices to shape [batch_size,F,L]
        # swap axes to shape [batch_size,L,F]
        # final shape [batch_size,L*F]
        output = torch.sum(torch.mul(weights, looked_up),
                         axis=-1).swapaxes(1, 2).reshape(x.shape[0], -1)
        return output





class my_NGP_INTERP_ENC(nn.Module):
    def __init__(self, geodesic_weight, lat_shape, lon_shape, n_levels=4, n_features_per_level=2, finest_resolution=0.25, base_resolution=1.00):
        # finest_resolution : 0.25의 역수 : 4
        # base_resolution (coarsest) : 1.00의 역수 : 1
        # b = 1.4142
        # N_level = b^level * N_min (주어진 range를 몇개의 grid로 쪼갤 것인지, 크면 클수록 fine resolution)
        # level=0 : coarsest
        # 
        
        super(my_NGP_INTERP_ENC, self).__init__()
        

        self.geodesic_weight = geodesic_weight
        self.lat_shape = lat_shape # latitude shape of the finest resolution
        self.lon_shape = lon_shape # longitude shape of the finest resolution
        self.n_levels = n_levels   # levels
        self.n_features_per_level = n_features_per_level # 
        self.finest_resolution = finest_resolution
        self.base_resolution = base_resolution
        # self.b = torch.exp((torch.log(1/torch.tensor(self.finest_resolution))-torch.log(1/torch.tensor(self.base_resolution)))/(n_levels-1))
        self.b=torch.tensor(2.0)
        self.bounding_box = torch.tensor([[LAT_MIN-finest_resolution, LON_MIN-finest_resolution],[LAT_MAX+finest_resolution, LON_MAX+finest_resolution]])# torch.empty((2, 2)) # left-upper-corner (min_lat_lon), right-bottom-corner(max_lat_lon)
        self.embeddings = nn.ModuleList([nn.Embedding((self.lat_shape+1)*(self.lon_shape+1),
                                        self.n_features_per_level) for i in range(n_levels)])
        # self.lat_shape+1, self.lon_shape+1을 해주는 이유는 각 grid의 "corner"에 learnable representation을 만들어줘야 하기 때문

        
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    
    def forward(self, x):
        # Input : [batch, 2]의 input coordinates
        # Output : [batch, representation_dim] 의 learnable inputs
        x_embedded_all = []
        for i in range(self.n_levels):
            # resolution = torch.floor((1/self.base_resolution) * self.b ** i) # i승만큼 늘린다는게,,, 2,3,4,,, 제곱을 해주는 숫자가 저렇게 늘어난다는건데.. 너무 확 늘어나는거 아닌가?
            grid_size_lat = self.base_resolution * self.b ** i
            grid_size_lon = grid_size_lat 
            grid_size = torch.tensor([grid_size_lat, grid_size_lon])
            grid_min_vertex, grid_max_vertex, keep_mask, grid_indices = self.get_grid_vertices(x, self.bounding_box, grid_size) # x : [B, 3], grid_min_vertex : [B, 3], grid_max_vertex : [B, 3]
            # 각 point의 정보가 담겨있는 x를 input으로 받아서, 각 point에 해당하는 grid의 min_vertex, max_vertex를 갖고온다.. 이 말인 즉슨, physical coordinate를 가져오겠다는 의미이다.
            
            # grid_indices : [batch, 4, 2] => 각 batch마다 input coordinates에 해당하는 4개의 corner의 index (이 index가 (a,b) 이런 식으로 2차원이면 안될 것 같음..)
            # self.embeddings[i](grid_indices) => [batch, 4, 2, 2]
            grid_embedds = self.embeddings[i](grid_indices.long())# (grid_min_vertex, grid_max_vertex에 해당하는 embeddings 정보를 가져와야 한다. 그런데 그 grid_min_vertex와 grid_max_vertex의 index 정보를 통해 가져와야 한다.) 

            grid_embedds = grid_embedds.to(x.device)
            x_embedded = self.bilinear_interp(x, grid_min_vertex, grid_max_vertex, grid_embedds)
            x_embedded_all.append(x_embedded)

        # keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        out = torch.cat(x_embedded_all, dim=-1)
        return out
            
            
            
    def get_grid_vertices(self, x, bounding_box, grid_size):
        # Input :  [batch, 2]의 좌표들이 들어오면
        # Output : [batch, 4, 2]의 해당 좌표를 감싸는 grid corners들의 index가 반환됨(grid_indices)
                  #[batch, 2]의 해당 좌표를 감싸는 grid corners의 min coords가 반환됨(grid_min_vertex)
                  #[batch, 2]의 해당 좌표를 감싸는 grid corners의 max coords가 반환됨(grid_max_vertex)
        '''
        theta_phi: latitude_longitude coordinates of samples. B x 2
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        '''
        bounding_box = bounding_box.to(x.device)
        # BOX_OFFSETS = BOX_OFFSETS.to(x.device)
        BOX_OFFSETS = torch.tensor([[i,j] for i in [0, 1] for j in [0, 1]],
                               device=x.device)

        grid_size = grid_size.to(x.device)
        # resolution = resolution.to(x.device)
        # lat_resolution = resolution
        # lon_resolution = resolution * 2
        # resolution = torch.tensor([lat_resolution, lon_resolution], device=x.device)
        
        
        
        box_min, box_max = bounding_box # bounding box의 physical min, max coordinate

        keep_mask = x==torch.max(torch.min(x, box_max), box_min) # xyz가 bounding box를 벗어나는지 확인
        if not torch.all(x <= box_max) or not torch.all(x >= box_min):
            # print("ALERT: some points are outside bounding box. Clipping them!")
            x = torch.clamp(x, min=box_min, max=box_max)

        # grid_size = (box_max-box_min)/resolution # bounding box 하나의 physical grid size
        
        bottom_left_idx = torch.floor((x-box_min)/grid_size).int() # 어떤 point x를 감싸는 grid의 index : [Batch, 2]
        grid_min_vertex = bottom_left_idx*grid_size + box_min # grid의 physical min coordinate
        # grid_max_vertex = grid_min_vertex + torch.tensor([1.0,1.0], device=x.device)*grid_size # grid의 physical max coordinate
        grid_max_vertex = grid_min_vertex + grid_size # 작은 grid의 (grid_min_vertex)다음 칸에 있는 grid

        # 3차원의 경우 voxel의 index 저장 [B, 8, 3]
        # 2차원의 경우 grid의  index 저장 [B, 4, 2]
        # bottom_left_idx를 갖고 있으면 거기서 나머지 bottom_right, up_left, up_right idx까지 BOX_OFFSETS로 구해주기
        twod_grid_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS  # [Batch, 4, 2] : 2d grid의 4개의 corner에 대한 index이므로 (x,y)형태로 저장되어져 있음. 
        
        # 2d matrix에 대한 index를 
        # 1d flatten vector의 index로 바꿔줌
        # width = lon_resolution# *(lon_resolution) # embedding의 width size
        # 1d_grid_indices = 
        # width = torch.floor(360/grid_size).int()+2 
        # width = # How shoud I set this??
        width = torch.floor(self.lon_shape / grid_size[...,1]).int()
            
        x_indices = twod_grid_indices[...,0]
        y_indices = twod_grid_indices[...,1]
        
        oned_grid_indices = x_indices * width + y_indices
        
        return grid_min_vertex, grid_max_vertex, keep_mask, oned_grid_indices

        

    # def bilinear_interp(self, grid, x, resolution):
    def bilinear_interp(self, x, grid_min_vertex, grid_max_vertex, grid_embedds):
        
        grid_min_vertex.to(x.device) # x를 감싸고 있는 grid의 left lower corner의 physical coordinate  # Used for weight
        grid_max_vertex.to(x.device) # x를 감싸고 있는 grid의 right above corner의 physical coordinate # Used for weight
        grid_embedds.to(x.device)    # x를 감싸고 있는 grid의 embedding 값                             # Used for interpolating target
        
        # Calculate weights for bilinear interpolation
        # (for x (latitude) and y (longitude) dimension)
        
        x_lat = x[:, 0]
        x_lon = x[:, 1]
        
        grid_max_lat = grid_max_vertex[:, 0]
        grid_max_lon = grid_max_vertex[:, 1]
        
        grid_min_lat = grid_min_vertex[:, 0]
        grid_min_lon = grid_min_vertex[:, 1]
        
        if self.geodesic_weight:
        # Calculate geodesic distances for weights
            weights_lat = self.geodesic(grid_min_lat, grid_min_lon, x_lat, grid_min_lon) / self.geodesic(grid_min_lat, grid_min_lon, grid_max_lat, grid_min_lon)
            weights_lon = self.geodesic(grid_min_lat, x_lon, grid_min_lat, grid_max_lon) / self.geodesic(grid_min_lat, grid_min_lon, grid_min_lat, grid_max_lon)

            # Combine weights for both dimensions
            weights = torch.stack([weights_lat, weights_lon], dim=1)
        else:
            weights = (x - grid_min_vertex)/(grid_max_vertex-grid_min_vertex)
        
        # Interpolation in x dimension
        c0 = grid_embedds[:, 0] * (1 - weights[:, 0][:, None]) + grid_embedds[:, 2] * weights[:, 0][:, None]
        c1 = grid_embedds[:, 1] * (1 - weights[:, 0][:, None]) + grid_embedds[:, 3] * weights[:, 0][:, None]

        # Interpolation in y dimension
        c = c0 * (1 - weights[:, 1][:, None]) + c1 * weights[:, 1][:, None]

        return c
    
    def geodesic(self, lat1, lon1, lat2, lon2):
        # R = 6371.0  # Approximate radius of earth in km

        rad_lat1 = torch.deg2rad(lat1)
        rad_lon1 = torch.deg2rad(lon1)
        rad_lat2 = torch.deg2rad(lat2)
        rad_lon2 = torch.deg2rad(lon2)

        dlon = rad_lon2 - rad_lon1
        dlat = rad_lat2 - rad_lat1

        a = torch.sin(dlat / 2)**2 + torch.cos(rad_lat1) * torch.cos(rad_lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        # distance = R * c
        return c

    '''
    def get_geodesic_weight(self, x, grid_min_vertex, grid_max_vertex):
        numerator = self.geodesic(x, grid_min_vertex)
        denominator = self.geodesic(grid_max_vertex, grid_min_vertex)
        
        out = numerator / denominator
        return out
    '''

    # def bilinear_interp(self, grid, x, resolution):
    # def geodesic_bilinear_interp(self, x, grid_min_vertex, grid_max_vertex, grid_embedds):
        
    #     grid_min_vertex.to(x.device) # x를 감싸고 있는 grid의 left lower corner의 physical coordinate  # Used for weight
    #     grid_max_vertex.to(x.device) # x를 감싸고 있는 grid의 right above corner의 physical coordinate # Used for weight
    #     grid_embedds.to(x.device)    # x를 감싸고 있는 grid의 embedding 값                             # Used for interpolating target
        
    #     # Calculate weights for bilinear interpolation
    #     # (for x (latitude) and y (longitude) dimension)
    #     # weights = (x - grid_min_vertex)/(grid_max_vertex-grid_min_vertex)
    #     weights = self.get_geodesic_weights(x, grid_min_vertex, grid_max_vertex)
        
    #     # Interpolation in x dimension
    #     c0 = grid_embedds[:, 0] * (1 - weights[:, 0][:, None]) + grid_embedds[:, 2] * weights[:, 0][:, None]
    #     c1 = grid_embedds[:, 1] * (1 - weights[:, 0][:, None]) + grid_embedds[:, 3] * weights[:, 0][:, None]

    #     # Interpolation in y dimension
    #     c = c0 * (1 - weights[:, 1][:, None]) + c1 * weights[:, 1][:, None]

    #     return c
        



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