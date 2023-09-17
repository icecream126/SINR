import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset
# import cv2
import matplotlib.pyplot as plt

from utils.utils import to_cartesian, add_noise, image_psnr


import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian, add_noise
TAU = 3e1 # Photon noise
NOISE_SNR = 2

class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        output_dim,
        normalize,
        panorama_idx,
        tau,
        snr,
        **kwargs
    ):
        self.dataset_dir = dataset_dir
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
        data_out['input_psnr'] = self._data['input_psnr']
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

        mean_lat_weight = torch.abs(torch.cos(lat)).mean()  # 0.6354

        lat, lon = torch.meshgrid(lat, lon)  # [2048, 1024]for each
        lat = lat.flatten()
        lon = lon.flatten()
        weights = torch.abs(torch.cos(lat)) / mean_lat_weight
        input_psnr = image_psnr(g_target, noisy_target, weights.reshape(g_target.shape[:2]).unsqueeze(-1).detach().cpu().numpy())
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
        data_out['input_psnr'] = input_psnr
        return data_out
    
'''
class Dataset(Dataset):
    def __init__(self, dataset_dir, output_dim, panorama_idx, tau, snr, batch_size, **kwargs):
        self.batch_size = batch_size
        self.output_dim = output_dim
        filetype = (
            ".jpg" if "sun" in dataset_dir else ".png"
        )  # sun360 - .jpg / flickr360 - .png
        im = normalize(
            plt.imread(dataset_dir + "/" + str(panorama_idx) + filetype).astype(
                np.float32
            ),
            True,
        )
        # im = cv2.resize(im, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
        self.H, self.W, _ = im.shape
        self.target_shape = im.shape

        # Create a noisy image
        im_noisy = measure(im, snr, tau)

        lat = np.linspace(-90, 90, self.H)
        lon = np.linspace(-180, 180, self.W)
        lat = torch.from_numpy(np.deg2rad(lat)).float()
        lon = torch.from_numpy(np.deg2rad(lon)).float()

        self.mean_lat_weight = torch.cos(lat).mean()

        lat, lon = torch.meshgrid(lat, lon)
        lat = lat.flatten()
        lon = lon.flatten()
        # weights = torch.cos(lat) / self.mean_lat_weight
        # w_psnr_val = psnr(im, im_noisy,weights.reshape(self.H, self.W).unsqueeze(-1).numpy())
        # print('Input Weighted PSNR : ',w_psnr_val)

        self.coords = torch.stack([lat, lon], dim=-1)

        self.gt = torch.from_numpy(im)
        self.gt_noisy = torch.from_numpy(im_noisy)

        self.gt = self.gt.reshape(-1, self.output_dim)
        self.gt_noisy = self.gt_noisy.reshape(-1, self.output_dim)

    def __len__(self):
        return 1
        # return len(self.coords) // self.batch_size
        # return self.["inputs"].size(0)

    def get_filenames(self):
        filenames = sorted(glob.glob(os.path.join(self.dataset_dir, "*")))
        return filenames[self.panorama_idx]

    def __getitem__(self, index):
        data_out = dict()
        
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        
        data_out["inputs"] = self.coords[start_idx : end_idx]
        data_out["target"] = self.gt_noisy[start_idx : end_idx]
        data_out["g_target"] = self.gt[start_idx : end_idx]
        data_out["target_shape"] = self.target_shape
        data_out["mean_lat_weight"] = self.mean_lat_weight
        return data_out
    
    def get_all_data(self):
        data_out = dict()
        data_out["inputs"] = self.coords
        data_out["target"] = self.gt_noisy
        data_out["g_target"] = self.gt
        data_out["target_shape"] = self.target_shape
        data_out["mean_lat_weight"] = self.mean_lat_weight
        return data_out
'''

'''
# This setting is to follow wire image denoising setting
# However, this shows poor performance than batch training
class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        output_dim,
        panorama_idx,
        tau,
        snr,
        **kwargs
    ):
        self.output_dim = output_dim
        filetype = '.jpg' if 'sun' in dataset_dir else '.png' # sun360 - .jpg / flickr360 - .png
        im = normalize(plt.imread(dataset_dir +'/'+ str(panorama_idx)+filetype).astype(np.float32),True)
        # im = cv2.resize(im, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
        self.H, self.W, _ = im.shape
        self.target_shape = im.shape
        
        # Create a noisy image
        im_noisy = measure(im, snr, tau)
        
        
        lat = np.linspace(-90, 90,self.H)
        lon = np.linspace(-180, 180,self.W)
        lat = torch.from_numpy(np.deg2rad(lat)).float()
        lon = torch.from_numpy(np.deg2rad(lon)).float()
        
        self.mean_lat_weight = torch.cos(lat).mean()
        
        lat, lon = torch.meshgrid(lat, lon)
        lat = lat.flatten()
        lon = lon.flatten()
        weights = torch.cos(lat) / self.mean_lat_weight
        # print('Input PSNR: %.2f dB'%psnr(im, im_noisy,weights.reshape(self.H, self.W).unsqueeze(-1).numpy()))
        
        error = torch.sum((torch.from_numpy(im)-torch.from_numpy(im_noisy))**2, dim=-1, keepdim=True)
        error = weights.reshape(self.H, self.W).unsqueeze(-1)*error
        loss = error.mean()
        w_psnr_val = psnr(im, loss.detach().cpu().numpy())
        print('Input Weighted PSNR : ',w_psnr_val)
        
        self.coords = torch.stack([lat, lon], dim=-1)
        
        
        self.gt = torch.from_numpy(im)
        self.gt_noisy = torch.from_numpy(im_noisy)
        
        self.gt = self.gt.reshape(-1, self.output_dim)
        self.gt_noisy = self.gt_noisy.reshape(-1, self.output_dim)
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        data_out = dict()
        data_out["inputs"] = self.coords
        data_out["target"] = self.gt_noisy
        data_out["g_target"] = self.gt
        data_out["target_shape"] = self.target_shape
        data_out["mean_lat_weight"] = self.mean_lat_weight
        
        return data_out
        

'''