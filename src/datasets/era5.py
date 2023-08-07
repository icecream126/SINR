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
            spherical=False,
            transform=None, 
            normalize=True, 
            **kwargs
        ):

        self.target_dim = 1

        self.spherical = spherical
        self.transform = transform
        self.normalize = normalize
        self.filepaths = self.get_filenames(dataset_dir, dataset_type)

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature
        data = np.load(self.filepaths[index])
        data_out = dict()

        temperature = torch.tensor(data['temperature'])  # Shape (num_lats, num_lons)
        latitude = data['latitude']  # Shape (num_lats,) theta
        longitude = data['longitude']  # Shape (num_lons,) phi

        lat_rad = self.deg_to_rad(latitude)
        lon_rad = self.deg_to_rad(longitude)

        if self.normalize:
            temperature = (temperature - T_MIN) / (T_MAX - T_MIN)

        longitude_grid, latitude_grid = np.meshgrid(lon_rad, lat_rad)

        latitude = torch.flatten(torch.tensor(latitude_grid))
        longitude = torch.flatten(torch.tensor(longitude_grid))
        # lat_rad = torch.tensor(lat_rad)
        # lon_rad = torch.tensor(lon_rad)

        if self.spherical:
            inputs = torch.stack([latitude, longitude], dim=-1)
        else:
            x = torch.cos(latitude) * torch.cos(longitude)
            y = torch.cos(latitude) * torch.sin(longitude)
            z = torch.sin(latitude)
            inputs = torch.stack([x, y, z], dim=-1)

        data_out["inputs"] = inputs
        data_out["target"] = torch.flatten(temperature)
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
    def get_filenames(dataset_dir, dataset_type):
        filenames = glob.glob(dataset_dir+'_'+dataset_type+'/*.npz')
        return sorted(filenames)