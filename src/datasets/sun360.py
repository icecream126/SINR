import os
import wandb
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from utils.change_coord_sys import to_cartesian

class SUN360(Dataset):
    
    train_sampled_indices = None
    
    def __init__(
            self,
            dataset,
            dataset_type, 
            sample_fraction=0.25, 
            panorama_idx=0, 
            spherical=False, 
            **kwargs
        ):
        super(SUN360, self).__init__()
        self.target_dim = 3
        
        self.dataset_dir = './dataset/' + dataset
        self.sample_fraction = sample_fraction        
        self.dataset_type = dataset_type
        self.panorama_idx = panorama_idx
        self.spherical = spherical
        self.panorama = [self.load_panorama(self.dataset_dir, panorama_idx)]
        self.sampled_indices = self._get_sampled_indices()

    def load_panorama(self, directory, panorama_idx):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image_file = os.listdir(directory)[panorama_idx]  # Load the panorama_idx image
        image_path = os.path.join(directory, image_file)
        wandb.log({"Ground Truth": wandb.Image(image_path)})
        panorama = transform(Image.open(image_path))
        return panorama


    def _get_sampled_indices(self):
        H, W = self.panorama[0].shape[1:]
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
        return len(self.panorama)

    def __getitem__(self, index):
        panorama = self.panorama[index]

        row = self.sampled_indices // panorama.shape[2] # Range : [0, height-1]
        col = self.sampled_indices % panorama.shape[2] # Range : [0, weight -1]
        
        # Convert pixel position to spherical coordinates
        theta = row * (np.pi/panorama.shape[1])  # Range : [0 , \pi]
        phi = col * (2*np.pi/panorama.shape[2])  # Range : [0, 2\pi]

        theta = torch.tensor(theta, dtype=torch.float32) - np.pi/2
        phi = torch.tensor(phi, dtype=torch.float32) - np.pi
        
        inputs = torch.stack([theta, phi], dim=-1)
        
        if not self.spherical:
            inputs = to_cartesian(inputs)

        rgb_value = panorama[:, row, col].transpose(0, 1)
        
        data_out = dict()
        data_out["inputs"] = inputs
        data_out["target"] = rgb_value
        return data_out