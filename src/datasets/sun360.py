import os
import wandb
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from utils.change_coord_sys import to_cartesian

class SUN360(Dataset):
    def __init__(
            self,
            dataset,
            dataset_type, 
            panorama_idx=0, 
            spherical=False,
            train_ratio=0.25,
            **kwargs
        ):
        super(SUN360, self).__init__()        
        self.dataset_dir = './dataset/' + dataset
        self.train_ratio = train_ratio
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
        panorama = transform(Image.open(image_path))
        
        wandb.log({"Ground Truth": wandb.Image(image_path)})
        return panorama


    def _get_sampled_indices(self):
        H, W = self.panorama[0].shape[1:]
        latitudes = np.linspace(-np.pi/2, np.pi/2, H)
        weights = np.cos(latitudes)
        weight_map = np.tile(weights[:, np.newaxis], (1, W))
        probabilities = weight_map.flatten() / np.sum(weight_map)
        
        total_size = H * W
        train_size = int(self.train_ratio * total_size)
        valid_size = test_size = int(0.25 * total_size)
        sample_size = train_size + valid_size + test_size

        if sample_size > total_size:
            raise ValueError('Decrease train ratio')
        
        np.random.seed(0)
        indice = np.random.choice(total_size, sample_size, replace=False, p=probabilities)

        idx_dict = {
            'train': indice[:train_size],
            'valid': indice[train_size:train_size+valid_size],
            'test': indice[train_size+valid_size:]
        }
        return idx_dict[self.dataset_type]

    def __len__(self):
        return len(self.panorama)

    def __getitem__(self, index):
        panorama = self.panorama[index]

        row = self.sampled_indices // panorama.shape[2] # Range : [0, height-1]
        col = self.sampled_indices % panorama.shape[2] # Range : [0, weight -1]
        
        # Convert pixel position to spherical coordinates
        theta = row * (np.pi/panorama.shape[1])  # Range : [0 , \pi]
        phi = col * (2*np.pi/panorama.shape[2])  # Range : [0, 2\pi]

        theta = torch.from_numpy(theta).float() - np.pi/2
        phi = torch.from_numpy(phi).float() - np.pi
        
        inputs = torch.stack([theta, phi], dim=-1)
        
        if not self.spherical:
            inputs = to_cartesian(inputs)

        rgb_value = panorama[:, row, col].transpose(0, 1)
        
        data_out = dict()
        data_out["inputs"] = inputs
        data_out["target"] = rgb_value
        return data_out