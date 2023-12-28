import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import exposure 

class NpyDataset(Dataset):
    def __init__(self, maxar_folder_irr, planet_folder_irr, maxar_folder_urban, planet_folder_urban, maxar_folder_snow, planet_folder_snow, train, inference, start, end, transform1=None, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        self.inference = inference
        self.maxar_folder_irr = maxar_folder_irr
        self.planet_folder_irr = planet_folder_irr
        self.maxar_folder_urban = maxar_folder_urban
        self.planet_folder_urban = planet_folder_urban
        self.maxar_folder_snow = maxar_folder_snow
        self.planet_folder_snow = planet_folder_snow
        self.maxar_files_complete = []
        self.end=end
        self.start=start
        self._load_tiles()
        


    def __len__(self):
        return len(self.maxar_files_complete)

    def __getitem__(self, idx):
        maxar_file = self.maxar_files_complete[idx]

        if "irrigation" in maxar_file:
            planet_folder = self.planet_folder_irr
        
        if "urban" in maxar_file:
            planet_folder = self.planet_folder_urban
        
        if "snow" in maxar_file:
            planet_folder = self.planet_folder_snow

        maxar_data = np.load(maxar_file)
        # Extract the corresponding Planet file name based on the Maxar file name
        filename = os.path.basename(maxar_file)
        
        planet_file = os.path.join(planet_folder, filename)
        planet_data = np.load(planet_file)

        if self.train:
            # noramlize_maxar = normalize(maxar_data)
            # noramlize_planet = normalize(planet_data)
            maxar_data1 = self.transform1(torch.tensor((maxar_data)))
            maxar_data2 = self.transform2(torch.tensor((maxar_data)))
            planet_data1 = self.transform1(torch.tensor(planet_data))
            planet_data2 = self.transform2(torch.tensor(planet_data))
            return maxar_data1, maxar_data2, planet_data1, planet_data2

        if self.inference:
            return self.transform1(torch.tensor(maxar_data)), self.transform2(torch.tensor(planet_data)), filename

    def _load_tiles(self):
        self.maxar_files_complete.extend(glob.glob(os.path.join(self.maxar_folder_irr, '*.npy'))[self.start:self.end])
        self.maxar_files_complete.extend(glob.glob(os.path.join(self.maxar_folder_urban, '*.npy'))[self.start:self.end])
        self.maxar_files_complete.extend(glob.glob(os.path.join(self.maxar_folder_snow, '*.npy'))[self.start:self.end])
            
# def min_max_scale_band(band):
#     min_value = np.min(band)
#     max_value = np.max(band)
    
#     # Check if the range is zero
#     if max_value == min_value:
#         # Handle zero range (you can choose an appropriate epsilon value)
#         epsilon = 1e-9
#         normalized_band = (band - min_value) / (max_value - min_value + epsilon)
#     else:
#         normalized_band = (band - min_value) / (max_value - min_value)
    
#     return normalized_band

# def normalize(image):
#     normalized_image = np.zeros_like(image, dtype=np.float32)
#     for band_idx in range(image.shape[0]):
#         normalized_image[band_idx:, :, ] = min_max_scale_band(image[band_idx:, :, ])
#     return normalized_image

# def normalize(image):
#     normalized_image = np.zeros_like(image)
#     for band_idx in range(image.shape[2]):
#         normalized_image[:, :, band_idx] = exposure.rescale_intensity(image[:, :, band_idx], in_range='image', out_range=(0, 1))
#     return normalized_image

# def normalize(image):
#     return exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))

