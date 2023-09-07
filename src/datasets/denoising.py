import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian, add_noise, image_psnr


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        output_dim,
        normalize,
        panorama_idx,
        tau,
        snr,
        **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.filename = self.get_filenames()
        self.tau = tau
        self.snr = snr
        self._data = self.load_data()

    def __len__(self):
        return self._data["inputs"].size(0)

    def get_filenames(self):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*"))
        return filenames[self.panorama_idx]

    def __getitem__(self, index):
        data_out = dict()

        data_out["inputs"] = self._data["inputs"][index]
        data_out["target"] = self._data["target"][index]
        data_out["g_target"] = self._data["g_target"][index]
        data_out["target_shape"] = self._data["target_shape"]
        data_out["mean_lat_weight"] = self._data["mean_lat_weight"]
        return data_out

    def load_data(self):
        data_out = dict()
        if "360" in self.dataset_dir:
            g_target = np.array(Image.open(self.filename))  # [1024, 2048, 3]

            target_shape = g_target.shape
            H, W = g_target.shape[:2]  # H : 1024, W : 2048

            lat = np.linspace(-90, 90, H)  # 1024
            lon = np.linspace(-180, 180, W)  # 2048
        # Circle
        else:
            data = np.load(self.filename)
            lat = data["latitude"]
            lon = data["longitude"]
            g_target = data["target"]  # [720, 1440]

            target_shape = g_target.shape

        if self.normalize:
            g_target = (g_target - g_target.min()) / (g_target.max() - g_target.min())
        noisy_target = add_noise(g_target, self.snr, self.tau)

        lat = torch.from_numpy(np.deg2rad(lat)).float()
        lon = torch.from_numpy(np.deg2rad(lon)).float()

        mean_lat_weight = torch.cos(lat).mean()  # 0.6354

        lat, lon = torch.meshgrid(lat, lon)  # [2048, 1024]for each
        lat = lat.flatten()
        lon = lon.flatten()
        weights = torch.cos(lat) / mean_lat_weight
        # coords = np.hstack((lat.reshape(-1,1), lon.reshape(-1,1)))  # [2097152, 2]
        inputs = torch.stack([lat, lon], dim=-1)
        # inputs = to_cartesian(inputs)  # [2097152, 3]
        # print('noisy img psnr : ',image_psnr(g_target, noisy_target, weights.numpy()))

        g_target = torch.from_numpy(g_target)
        noisy_target = torch.from_numpy(noisy_target)

        g_target = g_target.reshape(-1, self.output_dim)
        noisy_target = noisy_target.reshape(-1, self.output_dim)

        data_out["inputs"] = inputs  # input coordinate [2097152,3]
        data_out["target"] = noisy_target  # noisy image pixel value [2097152,3]
        data_out["g_target"] = g_target  # ground truth image pixel value [2097152,3]
        data_out["target_shape"] = target_shape  # 2097152,3
        data_out["mean_lat_weight"] = mean_lat_weight  # 0.6360
        return data_out
