import wandb
import torch
import numpy as np
from math import pi
from torch.utils.data import Dataset


class CIRCLE(Dataset):
    def __init__(
            self, 
            dataset_type,
            spherical=False,
            **kwargs
        ):

        self.target_dim = 1

        self.dataset_type = dataset_type
        self.spherical = spherical
        self._data = [self.generate_data(dataset_type)]

    def __getitem__(self, index):
        data_out = {}
        data_out['inputs'] = self._data[index]['inputs']
        data_out['target'] = self._data[index]['target']
        return data_out

    def __len__(self):
        return len(self._data)
    
    @staticmethod
    def deg_to_rad(degrees):
        return pi * degrees / 180.

    def generate_data(self, dataset_type, full_res=0.25):
        lat = np.arange(-90., 90., full_res)
        lon = np.arange(-180., 180., full_res)

        lat_rad = self.deg_to_rad(lat)
        lon_rad = self.deg_to_rad(lon)

        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        target = ((lat_grid>-pi/4) & (lat_grid<pi/4)).astype(float)
        wandb.log({"Ground Truth": wandb.Image(target)})

        lat = torch.flatten(torch.tensor(lat_grid, dtype=torch.float32))
        lon = torch.flatten(torch.tensor(lon_grid, dtype=torch.float32))
        target = torch.flatten(torch.tensor(target, dtype=torch.float32))

        if self.spherical:
            inputs = torch.stack([lat, lon], dim=-1)
        else:
            x = torch.cos(lat) * torch.cos(lon)
            y = torch.cos(lat) * torch.sin(lon)
            z = torch.sin(lat)
            inputs = torch.stack([x, y, z], dim=-1)

        indice = np.random.RandomState(seed=0).permutation(len(lat))

        train_size = 100000
        valid_size = 400000
        test_size = 400000

        idx_dict = {
            'train': indice[:train_size],
            'valid': indice[train_size:train_size+valid_size],
            'test': indice[train_size+valid_size:train_size+valid_size+test_size]
        }
        
        indice = idx_dict[dataset_type]
        
        data = {}
        data['inputs'] = inputs[indice]
        data['target'] = target[indice].unsqueeze(-1)
        return data

