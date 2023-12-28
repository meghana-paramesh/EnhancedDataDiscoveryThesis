import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import exposure 


class NpyDataset(Dataset):
    def __init__(self, folder_irr, folder_urban, folder_snow, train, inference, start, end, transform1=None, transform2=None):
        self.train = train
        self.inference = inference
        self.transform1 = transform1
        self.transform2 = transform2
        self.folder_irr = folder_irr
        self.folder_urban = folder_urban
        self.folder_snow = folder_snow
        self.complete_files = []
        self.end=end
        self.start=start

        self._load_all_files()

    def __len__(self):
        return len(self.complete_files)

    def __getitem__(self, idx):
        file = self.complete_files[idx]
        data = np.load(file)

        if self.train and self.transform1 and self.transform2:
            # normalized_data = norm(data)
            data1 = self.transform1(torch.tensor(data))
            data2 = self.transform2(torch.tensor(data))
            return data1, data2
        
        if self.inference:
            return self.transform1(torch.tensor(data)), self.transform2(torch.tensor(data)), os.path.basename(file)

    def _load_all_files(self):
        self.complete_files.extend(glob.glob(os.path.join(self.folder_irr, '*.npy'))[self.start:self.end])
        self.complete_files.extend(glob.glob(os.path.join(self.folder_urban, '*.npy'))[self.start:self.end])
        self.complete_files.extend(glob.glob(os.path.join(self.folder_snow, '*.npy'))[self.start:self.end])


# def normalize_image(image):
#     normalized_image = np.zeros_like(image)
#     for band_idx in range(image.shape[2]):
#         normalized_image[:, :, band_idx] = exposure.rescale_intensity(image[:, :, band_idx], in_range='image', out_range=(0, 1))
#     return normalized_image

class SingleDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, '*.npy'))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file)

        normalized_data = norm(data)
        # data1 = self.transform(torch.tensor(normalized_data))
        return normalized_data

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

# def norm(image):
#     normalized_image = np.zeros_like(image, dtype=np.float32)
#     for band_idx in range(image.shape[0]):
#         normalized_image[band_idx:, :, ] = min_max_scale_band(image[band_idx:, :, ])
#     return normalized_image
        


