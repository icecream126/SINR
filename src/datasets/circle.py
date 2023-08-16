import torch
import numpy as np
from math import pi
from torch.utils.data import Dataset

from utils.change_coord_sys import to_cartesian


class CIRCLE(Dataset):
    def __init__(
            self,
            dataset_type,
            spherical=False,
            **kwargs
        ):
        self.dataset_type = dataset_type
        self.spherical = spherical
        self._data = [self.generate_data()]

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

    def generate_data(self, res=0.25):

        lat = np.arange(-90, 90, res)
        lon = np.arange(-180, 180, res)

        lat = self.deg_to_rad(lat)
        lon = self.deg_to_rad(lon)

        if self.dataset_type == 'train':
            start = 0 
        elif self.dataset_type == 'valid':
            start = 1
        else:
            start = 2

        lat_idx = np.arange(start, len(lat), 3)
        lon_idx = np.arange(start, len(lon), 3)

        lat = torch.from_numpy(lat[lat_idx]).float()
        lon = torch.from_numpy(lon[lon_idx]).float()

        lat, lon = torch.meshgrid(lat, lon)
        target = torch.logical_and(lat>-pi/4, lat<pi/4).float()

        lat = lat.flatten()
        lon = lon.flatten()
        target = target.flatten()

        inputs = torch.stack([lat, lon], dim=-1)
        
        if not self.spherical:
            inputs = to_cartesian(inputs)
        
        data = {}
        data['inputs'] = inputs
        data['target'] = target.unsqueeze(-1)
        return data

