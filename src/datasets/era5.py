import os
import glob
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset


T_MIN = 202.66
T_MAX = 320.93


class ERA5(Dataset):
    """ERA5 temperature dataset.

    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """
    def __init__(
            self, 
            dataset_dir,
            dataset_type,
            temporal_res,
            spatial_res,
            time,
            spherical=False,
            transform=None, 
            normalize=True, 
            **kwargs
        ):

        self.target_dim = 1

        self.dataset_type = dataset_type
        self.temporal_res = temporal_res
        self.spatial_res = spatial_res
        self.time = time
        self.spherical = spherical
        self.transform = transform
        self.normalize = normalize

        self.filepaths = self.get_filenames(dataset_dir)
        self.time_idx = self.get_time_idx(dataset_type, temporal_res, len(self.filepaths))
        self.filepaths = self.filepaths[self.time_idx]

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature

        data = np.load(self.filepaths[index])
        data_out = dict()

        time = torch.tensor(data['time'])
        temperature = torch.tensor(data['temperature'])  # Shape (num_lats, num_lons)
        latitude = data['latitude']  # Shape (num_lats,) theta
        longitude = data['longitude']  # Shape (num_lons,) phi

        full_res = 180/len(latitude)

        if self.dataset_type == 'train':
            step = int(self.spatial_res/full_res)
        else:
            step = int(self.spatial_res/full_res/2) # valid, test는 2배 해상도
        
        if step < 1:
            raise ValueError('Exceed full resolution. Increase spatial_res')

        lat_idx = np.arange(0, len(latitude), step)
        lon_idx = np.arange(0, len(longitude), step)

        latitude = latitude[lat_idx]
        longitude = longitude[lon_idx]
        temperature = temperature[lat_idx][:, lon_idx]

        lat_rad = self.deg_to_rad(latitude)
        lon_rad = self.deg_to_rad(longitude)

        if self.normalize:
            temperature = (temperature - T_MIN) / (T_MAX - T_MIN)

        longitude_grid, latitude_grid = np.meshgrid(lon_rad, lat_rad)

        latitude = torch.flatten(torch.tensor(latitude_grid))
        longitude = torch.flatten(torch.tensor(longitude_grid))
        time = time * torch.ones_like(latitude)

        if self.spherical:
            inputs = torch.stack([latitude, longitude, time], dim=-1)
        else:
            x = torch.cos(latitude) * torch.cos(longitude)
            y = torch.cos(latitude) * torch.sin(longitude)
            z = torch.sin(latitude)
            inputs = torch.stack([x, y, z, time], dim=-1)

        data_out["inputs"] = inputs
        data_out["target"] = torch.flatten(temperature).unsqueeze(-1)
        return data_out

        

        # Create a grid of latitude and longitude values matching the shape
        # of the temperature grid
        # longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
        # Shape (3, num_lats, num_lons)
        # data_tensor = np.stack([latitude_grid, longitude_grid, temperature])
        # data_tensor = torch.Tensor(data_tensor)
        # Perform optional transform
        # if self.transform:
        #     data_tensor = self.transform(data_tensor)
        # return data_tensor, 0  # Label to ensure consistency with image datasets

    def __len__(self):
        return len(self.filepaths)

    @staticmethod
    def deg_to_rad(degrees):
        return np.pi * degrees / 180.

    @staticmethod
    def get_filenames(dataset_dir):
        filenames = glob.glob(dataset_dir+'/*.npz')
        filenames = sorted(filenames)
        return np.array(filenames)

    @staticmethod
    def get_time_idx(dataset_type, temporal_res, length):
        indice = np.random.RandomState(seed=0).permutation(length)

        train_size = int(length/temporal_res)
        valid_size = int(2*length/temporal_res)
        test_size = int(2*length/temporal_res)

        if train_size + valid_size + test_size > length:
            raise ValueError('Exceed full resolution. Increase temporal_res')

        idx_dict = {
            'train': indice[:train_size],
            'valid': indice[train_size:train_size+valid_size],
            'test': indice[train_size+valid_size:train_size+valid_size+test_size]
        }
        return idx_dict[dataset_type]