import os
import glob

import torch
import numpy as np
from math import pi
from torch.utils.data import Dataset

TIME_MIN = 692496
TIME_MAX = 1043135

Z_MIN = 42481
Z_MAX = 59345

class ERA5(Dataset):
    def __init__(
            self, 
            dataset,
            dataset_type,
            spherical,
            sample_ratio,
            time_scale,
            z_scale,
            **kwargs
        ):

        self.dataset_path = 'dataset/' + dataset
        self.dataset_type = dataset_type
        self.spherical = spherical
        self.sample_ratio = sample_ratio
        self.time_scale = time_scale
        self.z_scale = z_scale

        self.filenames = self.get_filenames(self.dataset_path)

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature

        filename = self.filenames[index]
        data = np.load(filename)

        data_out = dict()

        lat = data['latitude']
        lon = data['longitude']
        time = data['time']
        target = data['z']
        
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

        lat_idx = np.random.permutation(lat_idx)[:lat_sample_num]
        lon_idx = np.random.permutation(lon_idx)[:lon_sample_num]
        time_idx = np.random.permutation(time_idx)[:time_sample_num]

        lat = torch.from_numpy(lat[lat_idx])
        lon = torch.from_numpy(lon[lon_idx])
        time = torch.from_numpy(time[time_idx])
        target = torch.from_numpy(target[time_idx][:, lat_idx][:, :, lon_idx]).float()

        lat = self.deg_to_rad(lat)
        lon = self.deg_to_rad(lon)
        time = (time - TIME_MIN) / self.time_scale # (TIME_MAX - TIME_MIN)
        target = (target - Z_MIN) / self.z_scale # (Z_MAX - Z_MIN)

        time, lat, lon = torch.meshgrid(time, lat, lon)

        lat = lat.flatten()   
        lon = lon.flatten()   
        time = time.flatten()   
        target = target.flatten()

        if self.spherical:
            inputs = torch.stack([lat, lon, time], dim=-1)
        else:
            x = torch.cos(lat) * torch.cos(lon)
            y = torch.cos(lat) * torch.sin(lon)
            z = torch.sin(lat)
            inputs = torch.stack([x, y, z, time], dim=-1)

        data_out['inputs'] = inputs
        data_out['target'] = target.unsqueeze(-1)
        return data_out

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def deg_to_rad(degrees):
        return pi * degrees / 180.

    @staticmethod
    def get_filenames(dataset_dir):
        filenames = glob.glob(os.path.join(dataset_dir, "*.npz"))
        return sorted(filenames)