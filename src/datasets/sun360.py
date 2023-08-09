import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

np.random.seed(1234) # TODO : get seed as args

class SUN360(Dataset):
    
    train_sampled_indices = None
    
    def __init__(self, dataset_dir, dataset_type, sample_fraction=0.25, panorama_idx=0, spherical=False, **kwargs):
        super(SUN360, self).__init__()
        self.sample_fraction = sample_fraction        
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.panorama_idx = panorama_idx
        self.spherical = spherical
        self.panorama = self.load_panorama(self.dataset_dir, self.panorama_idx)
        self.sampled_indices = self._get_sampled_indices()
        self.target_dim = 3

    def load_panorama(self, directory, panorama_idx):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image_file = os.listdir(directory)[panorama_idx]  # Load the panorama_idx image
        image_path = os.path.join(directory, image_file)
        panorama = transform(Image.open(image_path))
        
        return panorama


    def _get_sampled_indices(self):
        H, W = self.panorama.shape[1:3]
        latitudes = np.linspace(-np.pi/2, np.pi/2, H)
        weights = np.cos(latitudes)
        weight_map = np.tile(weights[:, np.newaxis], (1, W))
        probabilities = weight_map.flatten() / np.sum(weight_map)
        
        total_pixels = H * W
        num_pixels_to_sample = int(self.sample_fraction * total_pixels)
        
        if self.dataset_type == 'train':
            sampled_indices = np.random.choice(total_pixels, num_pixels_to_sample, replace=False, p=probabilities)
            SUN360.train_sampled_indices = sampled_indices
        else:  # 'test'
            all_indices = np.arange(total_pixels)
            train_indices = SUN360.train_sampled_indices  # Train indices are stored in SUN360.train_sampled_indices
            test_indices = np.setdiff1d(all_indices, train_indices)
            sampled_indices = np.random.choice(test_indices, num_pixels_to_sample, replace=False)
        
        return sampled_indices

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        flat_idx = self.sampled_indices[idx]
        row = flat_idx // self.panorama.shape[2] # Range : [0, height-1]
        col = flat_idx % self.panorama.shape[2] # Range : [0, weight -1]
        
        # Convert pixel position to spherical coordinates
        theta = row * (np.pi/self.panorama.shape[1])  # Range : [0 , \pi]
        phi = col * (2*np.pi/self.panorama.shape[2])  # Range : [0, 2\pi]
        
        spherical_coords = torch.tensor([theta, phi], dtype=torch.float32)
        rgb_value = self.panorama[:, row, col]
        
        if self.spherical:
            inputs = spherical_coords
        else:
            theta = torch.tensor(theta)
            phi = torch.tensor(phi)
            x = torch.cos(theta) * torch.cos(phi)
            y = torch.cos(theta) * torch.sin(phi)
            z = torch.sin(theta)
            inputs = torch.tensor([x,y,z], dtype = torch.float32)
        
        data_out = dict()
        data_out["inputs"] = inputs
        data_out["target"] = rgb_value
        
        return data_out