# Code from : https://github.com/EmilienDupont/neural-function-distributions/blob/main/data/dataloaders_era5.py
from argparse import ArgumentParser

import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Statistics for the era5_temp2m_16x_train dataset (in Kelvin)
T_MIN = 202.66
T_MAX = 320.93


def era5(path_to_data, batch_size=16):
    """ERA5 climate data.

    Args:
        path_to_data (string):
        batch_size (int):
    """
    dataset = ERA5EuclideanDataset(path_to_data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
def deg_to_rad(degrees):
    return np.pi * degrees / 180.


class ERA5EuclideanDataset(Dataset):
    """ERA5 temperature dataset.

    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """
    def __init__(self, time=False, transform=None, normalize=True, dataset_type='train',**kwargs):
        self.time = False
        self.target_dim = 1
        self.transform = transform
        self.normalize = normalize
        self.filepaths = glob.glob('./dataset/era5_temp2m_16x/'+dataset_type+'/*.npz')
        self.filepaths.sort()  # Ensure consistent ordering of paths

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature
        data = np.load(self.filepaths[index])
        data_out = dict()

        temperature = torch.tensor(data['temperature'])  # Shape (num_lats, num_lons)
        latitude = data['latitude']  # Shape (num_lats,) theta
        longitude = data['longitude']  # Shape (num_lons,) phi

        lat_rad = deg_to_rad(latitude)
        lon_rad = deg_to_rad(longitude)



        if self.normalize:
            temperature = (temperature - T_MIN) / (T_MAX - T_MIN)

        longitude_grid, latitude_grid = np.meshgrid(lon_rad, lat_rad)

        latitude = torch.flatten(torch.tensor(latitude_grid))
        longitude = torch.flatten(torch.tensor(longitude_grid))
        # lat_rad = torch.tensor(lat_rad)
        # lon_rad = torch.tensor(lon_rad)
        x = torch.cos(latitude) * torch.cos(longitude)
        y = torch.cos(latitude) * torch.sin(longitude)
        z = torch.sin(latitude)
        
        inputs = torch.stack([x,y,z],dim=-1)

        data_out["inputs"] = inputs
        data_out["target"] = torch.flatten(temperature)
        data_out["index"] = index

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
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("—dataset_dir", type=str)
        parser.add_argument("—n_nodes_in_sample", type=int, default=10000)
        parser.add_argument("—time", type=bool, default=True)
        parser.add_argument("—in_memory", type=bool, default=True)

        return parser
