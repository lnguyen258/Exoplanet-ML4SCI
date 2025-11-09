import os
from astropy.io import fits
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DiskDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.fits')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        hdul = fits.open(file_path)
        image = hdul[0].data.astype(np.float32)
        image = image[0].squeeze() 
        hdul.close()

        image = self._normalize_to_uint8(image)
        pil_image = Image.fromarray(image, mode='L')
        return pil_image
    
    def _normalize_to_uint8(self, image):
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        img_min = image.min() 
        img_max = image.max()  
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)  
        image = (image * 255).astype(np.uint8)
        return image