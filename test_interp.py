import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


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
            latent_reps.append(latent_rep)

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
        lat_frac = (lat_idx - lat_floor).unsqueeze(-1)
        lon_frac = (lon_idx - lon_floor).unsqueeze(-1)

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
    

def test_learnable_encoding():
    # Initialize the LearnableEncoding instance
    lat_shape, lon_shape, level, resolution = 5, 8, 3, 45.0
    # latitude = -90, 90
    # longitude = 0, 360 (except 360)
    # latitude : [-90, -45, 0, 45, 90]
    # longitude : [0, 45, 90, 135, 180, 225, 270, 315]
    
    
    model = LearnableEncoding(lat_shape, lon_shape, level, resolution)

    # Generate random latitude and longitude values considering the resolution
    # num_samples = 5  # Number of samples to test
    # latitudes = np.random.uniform(-90, 90) / resolution
    # longitudes = np.random.uniform(0, 359) / resolution
    latitudes = torch.tensor([-90.0, -67.5, -45.0, -22.5,  0.0, 22.5, 45.0, 67.5, 90.0])
    longitudes = torch.tensor([0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0])
    
    latitudes, longitudes = torch.meshgrid(latitudes, longitudes)
    latitudes, longitudes = latitudes.flatten(), longitudes.flatten()

    # Round to the nearest grid point and then multiply by resolution
    # latitudes = np.round(latitudes) * resolution
    # longitudes = np.round(longitudes) * resolution

    # Convert to a PyTorch tensor
    coordinates = torch.tensor(torch.stack([latitudes, longitudes], dim=1), dtype=torch.float32)

    # Pass the coordinates through the model
    output = model(coordinates)

    print("Input Coordinates:\n", coordinates)
    print("Output from LearnableEncoding:\n", output)

# Run the test
test_learnable_encoding()