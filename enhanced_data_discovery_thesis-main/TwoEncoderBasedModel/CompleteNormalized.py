import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import exposure 


class NpyDataset(Dataset):
    def __init__(self, folder, train, inference, transform1=None, transform2=None):
        self.files = glob.glob(os.path.join(folder, '*.npy'))
        self.train = train
        self.inference = inference
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file)

        if self.train and self.transform1 and self.transform2:
            normalized_data = normalize(data)
            data1 = self.transform1(torch.tensor(normalized_data))
            data2 = self.transform2(torch.tensor(normalized_data))
            return data1, data2
        
        if self.inference:
            return normalize(data), normalize(data), os.path.basename(file)

#
# imagearray
# Image array.

# in_range, out_rangestr or 2-tuple, optional
# Min and max intensity values of input and output image. The possible values for this parameter are enumerated below.

# ‘image’
# Use image min/max as the intensity range.

# ‘dtype’
# Use min/max of the image’s dtype as the intensity range.

# dtype-name
# Use intensity range based on desired dtype. Must be valid key in DTYPE_RANGE.

# 2-tuple
# Use range_values as explicit min/max intensities.
def normalize(image):
    return exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
