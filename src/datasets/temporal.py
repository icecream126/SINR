import os
import glob

import torch
import numpy as np
from math import pi
import netCDF4 as nc
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
            self,
            dataset_path,
            dataset_type,
            output_dim,
            sample_ratio,
            **kwargs
        ):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.sample_ratio = sample_ratio
        self.filenames = self.get_filenames()

    def __getitem__(self, index):
        data_out = dict()

        if 'era5' in self.filenames[index]:
            with nc.Dataset(self.filenames[index], 'r') as f:
                for variable in f.variables:
                    if variable == 'latitude':
                        lat = f.variables[variable][:]
                    elif variable == 'longitude':
                        lon = f.variables[variable][:]
                    elif variable == 'time':
                        time = f.variables[variable][:]
                    elif variable in ['z', 't']:
                        target = f.variables[variable]

        lat = self.deg_to_rad(lat)
        lon = self.deg_to_rad(lon)
        time = (time - time.min()) / (time.max() - time.min())
        target = (target - target.min()) / (target.max() - target.min())
        
        if self.dataset_type == 'train':
            start = 0 
        elif self.dataset_type == 'valid':
            start = 1
        else:
            start = 2

        lat_idx = np.arange(start, len(lat), 3)
        lon_idx = np.arange(start, len(lon), 3)
        time_idx = np.arange(start, len(time), 3)

        lat_sample_num = int(len(lat_idx)*(self.sample_ratio)**(1/3))
        lon_sample_num = int(len(lon_idx)*(self.sample_ratio)**(1/3))
        time_sample_num = int(len(time_idx)*(self.sample_ratio)**(1/3))

        lat_idx = np.random.choice(lat_idx, lat_sample_num, replace=False)
        lon_idx = np.random.choice(lon_idx, lon_sample_num, replace=False)
        time_idx = np.random.choice(time_idx, time_sample_num, replace=False)

        lat = torch.from_numpy(lat[lat_idx]).float()
        lon = torch.from_numpy(lon[lon_idx]).float()
        time = torch.from_numpy(time[time_idx]).float()
        target = torch.from_numpy(target[time_idx][:, lat_idx][:, :, lon_idx]).float()

        time, lat, lon = torch.meshgrid(time, lat, lon)

        lat = lat.flatten()
        lon = lon.flatten()
        time = time.flatten()
        target = target.reshape(-1, self.output_dim)

        inputs = torch.stack([lat, lon, time], dim=-1)

        data_out['inputs'] = inputs
        data_out['target'] = target
        return data_out

    def __len__(self):
        return len(self.filenames)
    
    def get_filenames(self):
        filenames = glob.glob(os.path.join(self.dataset_path, "*.npz"))
        return sorted(filenames)

    @staticmethod
    def deg_to_rad(degrees):
        return pi * degrees / 180.
