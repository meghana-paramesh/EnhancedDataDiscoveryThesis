import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import exposure 
import random


class NpyDataset(Dataset):
    def __init__(self, folder_irr, folder_urban, folder_snow, train, inference, start, end, transform):
        self.train = train
        self.inference = inference
        self.folder_irr = folder_irr
        self.folder_urban = folder_urban
        self.folder_snow = folder_snow
        self.complete_files = []
        self.transform = transform
        self.start = start
        self.end = end

        self._load_all_files()

    def __len__(self):
        return len(self.complete_files)

    def __getitem__(self, idx):
        file = self.complete_files[idx]

        if("urban" in file):
            negative_sample = random.choice([cls for cls in self.complete_files if "urban" not in cls])
            positive_sample = random.choice([cls for cls in self.complete_files if "urban" in cls])
        
        if("irrigation" in file):
            negative_sample = random.choice([cls for cls in self.complete_files if "irrigation" not in cls])
            positive_sample = random.choice([cls for cls in self.complete_files if "irrigation" in cls])
        
        if("snow" in file):
            negative_sample = random.choice([cls for cls in self.complete_files if "snow" not in cls])
            positive_sample = random.choice([cls for cls in self.complete_files if "snow" in cls])
        data = np.load(file)

        if self.train and self.transform:
            anchor_image = np.load(file)
            positive_image = np.load(positive_sample)
            negative_image = np.load(negative_sample)

            return self.transform(norm(anchor_image)), self.transform(norm(positive_image)), self.transform(norm(negative_image))
        
        if self.inference:
            return self.transform(torch.tensor(norm(data))), os.path.basename(file)

    def _load_all_files(self):
        self.complete_files.extend(glob.glob(os.path.join(self.folder_irr, '*.npy'))[self.start:self.end])
        self.complete_files.extend(glob.glob(os.path.join(self.folder_urban, '*.npy'))[self.start:self.end])
        self.complete_files.extend(glob.glob(os.path.join(self.folder_snow, '*.npy'))[self.start:self.end])

def min_max_scale_band(band):
    min_value = np.min(band)
    max_value = np.max(band)
    
    # Check if the range is zero
    if max_value == min_value:
        # Handle zero range (you can choose an appropriate epsilon value)
        epsilon = 1e-9
        normalized_band = (band - min_value) / (max_value - min_value + epsilon)
    else:
        normalized_band = (band - min_value) / (max_value - min_value)
    
    return normalized_band

def norm(image):
    normalized_image = np.zeros_like(image, dtype=np.float32)
    for band_idx in range(image.shape[0]):
        normalized_image[band_idx:, :, ] = min_max_scale_band(image[band_idx:, :, ])
    return normalized_image