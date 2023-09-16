import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian


class Dataset(Dataset):
    def __init__(
        self, dataset_dir, dataset_type, output_dim, normalize, panorama_idx, **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.filename = self.get_filenames()
        self._data = self.load_data()

    def __len__(self):
        return self._data["inputs"].size(0)

    def get_filenames(self):
        filenames = sorted(glob.glob(os.path.join(self.dataset_dir, "*.nc")))
        return (
            filenames[self.panorama_idx] if "360" in self.dataset_dir else filenames[0]
        )

    def __getitem__(self, index):
        data_out = dict()

        data_out["inputs"] = self._data["inputs"][index]
        data_out["target"] = self._data["target"][index]
        data_out["target_shape"] = self._data["target_shape"]
        data_out["mean_lat_weight"] = self._data["mean_lat_weight"]
        data_out["lat"] = self._data["lat"]
        data_out["lon"] = self._data["lon"]
        # import pdb

        # pdb.set_trace()
        return data_out

    def load_data(self):
        data_out = dict()

        if "360" in self.dataset_dir:
            target = np.array(Image.open(self.filename))  # [512, 1024, 3]

            H, W = target.shape[:2]  # H : 512, W : 1024

            lat = np.linspace(-90, 90, H)  # 512
            lon = np.linspace(-180, 180, W)  # 1024

        elif "era5" in self.dataset_dir:
            with nc.Dataset(self.filename, "r") as f:
                for variable in f.variables:
                    if variable == "latitude":
                        lat = f.variables[variable][:]
                    elif variable == "longitude":
                        lon = f.variables[variable][:]
                    else:
                        target = f.variables[variable][0]
        else:
            data = np.load(self.filename)

            lat = data["latitude"]
            lon = data["longitude"]
            target = data["target"]

        lat = np.deg2rad(lat)  # 512 (min : -1.57, max : 1.57)
        lon = np.deg2rad(
            lon
        )  # 1024 (min : -3.14, max : 3.14) # ER5 (min : 0.0, max : 359.75)

        if self.normalize:
            target = (target - target.min()) / (target.max() - target.min())

        if self.dataset_type == "all":
            start, step = 0, 1
        elif self.dataset_type == "train":
            start, step = 0, 3
        elif self.dataset_type == "valid":
            start, step = 1, 3
        else:
            start, step = 2, 3

        lat_idx = np.arange(start, len(lat), step)
        lon_idx = np.arange(start, len(lon), step)

        lat = torch.from_numpy(lat[lat_idx]).float()  # 171
        lon = torch.from_numpy(lon[lon_idx]).float()  # 342
        target = torch.from_numpy(target[lat_idx][:, lon_idx]).float()  # [171, 342, 3]
        data_out["lat"] = lat
        data_out["lon"] = lon

        mean_lat_weight = torch.cos(lat).mean()  # 0.6341
        target_shape = target.shape  # [171, 342, 3]

        lat, lon = torch.meshgrid(lat, lon)  # [171, 342] for each

        lat = lat.flatten()  # [58482]
        lon = lon.flatten()  # ""
        target = target.reshape(-1, self.output_dim)  # [58482, 3]

        inputs = torch.stack([lat, lon], dim=-1)  # [58482, 2]
        # inputs = to_cartesian(inputs)  # [58482, 3]

        data_out["inputs"] = inputs  # [58482, 3]
        data_out["target"] = target  # [58482, 3]
        data_out["target_shape"] = target_shape  # [171, 342, 3]
        data_out["mean_lat_weight"] = mean_lat_weight  # 0.6341
        return data_out
