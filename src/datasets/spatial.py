import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset
import healpy as hp
import matplotlib.pyplot as plt
from utils.utils import gethealpixmap, img2healpix_planar, to_cartesian, StandardScalerTorch, MinMaxScalerTorch
import cv2
from mhealpy import HealpixMap

class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        input_dim,
        output_dim,
        normalize,
        panorama_idx,
        model,
        downscale_factor,
        zscore_normalize=False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.model = model
        self.downscale_factor = downscale_factor
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.zscore_normalize = zscore_normalize
        self.panorama_idx = panorama_idx
        self.scaler = None
        if self.normalize:
            self.scaler = MinMaxScalerTorch()
        elif self.zscore_normalize:
            self.scaler = StandardScalerTorch()
        self.filename = self.get_filenames()
        self._data = self.load_data()

    def __len__(self):
        return self._data["inputs"].size(0)

    def get_filenames(self):
        filenames = sorted(glob.glob(os.path.join(self.dataset_dir, "*")))
        if "sun360" in self.dataset_dir:
            return [
                filename
                for filename in filenames
                if f"{self.panorama_idx}.jpg" in filename
            ][0]
        elif "flickr360" in self.dataset_dir:
            return [
                filename
                for filename in filenames
                if f"{self.panorama_idx}.png" in filename
            ][0]
        else:
            return filenames[0]

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
            # if 'flickr' in self.dataset_dir:
            #     target = cv2.resize(target, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

            if self.dataset_type == "train":
                target = cv2.resize(
                    target,
                    None,
                    fx=1 / self.downscale_factor,
                    fy=1 / self.downscale_factor,
                    interpolation=cv2.INTER_AREA,
                )

            H, W = target.shape[:2]  # H : 512, W : 1024

            # if self.model=='ngp_interp' or 'coolchic_interp':
            if self.input_dim == 2:
                lat = torch.linspace(90, -90, H)
                lon = torch.linspace(0, 360, W)
            else:
                # lat = torch.linspace(-90, 90, H)  # 512
                # lon = torch.linspace(-180, 180, W)  # 1024
                lat = torch.linspace(90, -90, H)
                lon = torch.linspace(0, 360, W)

        elif "era5" in self.dataset_dir:
            if self.dataset_type == "train":
                parts = self.dataset_dir.split("/")
                if self.downscale_factor == 2:  # 0_50
                    parts[1] = "spatial_0_50"
                    filename = "/".join(parts) + "/data.nc"
                elif self.downscale_factor == 4:  # 1_00
                    parts[1] = "spatial_1_00"
                    filename = "/".join(parts) + "/data.nc"
                else:
                    raise Exception("Unsupported downscaling factor for ERA5")
            else:
                filename = self.filename

            with nc.Dataset(filename, "r") as f:
                for variable in f.variables:
                    if variable == "latitude":
                        lat = f.variables[variable][:]
                    elif variable == "longitude":
                        lon = f.variables[variable][:]
                    else:
                        target = f.variables[variable][0]

            target = torch.from_numpy(target)
            lat = torch.from_numpy(lat) # Adjust into colatitude range (0 : North Pole, 90 : Equator, 180 : South Pole)
            lon = torch.from_numpy(lon)
            
            
        else:
            data = np.load(self.filename)

            lat = data["latitude"]
            lon = data["longitude"]
            target = data["target"]

            if self.dataset_type == "train":
                target = cv2.resize(
                    target,
                    None,
                    fx=1 / self.downscale_factor,
                    fy=1 / self.downscale_factor,
                    interpolation=cv2.INTER_AREA,
                )
            H, W = target.shape[:2]  # H : 512, W : 1024

            lat = torch.linspace(90, -90, H)
            lon = torch.linspace(0, 360, W)

            
        # if self.model!='learnable' and self.model!='coolchic_interp' and self.model!='ngp_interp':
        if self.input_dim == 3:
            lat = torch.deg2rad(lat)  # 512 (min : -1.57, max : 1.57)
            lon = torch.deg2rad(
                lon
            )  # 1024 (min : -3.14, max : 3.14) # ER5 (min : 0.0, max : 359.75)

        data_out["lat"] = lat
        data_out["lon"] = lon

        mean_lat_weight = torch.abs(torch.cos(lat)).mean()  # 0.6341
        target_shape = target.shape  # [171, 342, 3]

        lat, lon = torch.meshgrid(lat, lon)  # [171, 342] for each
            
        lat = lat.flatten()  # [58482]
        lon = lon.flatten()  # ""
        target = target.reshape(-1, self.output_dim)  # [58482, 3]

        # HEALPix isn't for image
        # TODO : Finish multi-resolution map
        # if "360" not in self.dataset_dir:
        #     hp_map, m_hp_map = gethealpixmap(data = np.array(target.squeeze(-1)), nside = 64, theta = np.array(torch.deg2rad(lat)), phi = np.array(torch.deg2rad(lon)))

        if self.zscore_normalize or self.normalize:
            self.scaler.fit(target)
            target = self.scaler.transform(target)

        inputs = torch.stack([lat, lon], dim=-1)  # [58482, 2]
        # inputs = to_cartesian(inputs)  # [58482, 3]

        data_out["inputs"] = inputs  # [58482, 3]
        data_out["target"] = target  # [58482, 3]
        data_out["target_shape"] = target_shape  # [171, 342, 3]
        data_out["mean_lat_weight"] = mean_lat_weight  # 0.6341

        return data_out
