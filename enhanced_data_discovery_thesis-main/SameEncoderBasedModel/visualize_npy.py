import sqlite3
import faiss
import torch
import numpy as np
from natsort import natsorted
import os
from torchvision import models
from ImageDataset import ImageDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

def visualize():
    data_img = "tiles_all_new/desert_217_409_maxar.npy"
    original_type="MAXAR"
    data = np.load(data_img)
    print(data.shape)
    # if original_type=="PLANET":
    #     selected_bands = [2, 4, 6]
    #     # data = data / 255.0
    #     rgb_bands_data = data[selected_bands, :, :]
    #     plt.imshow(np.transpose(rgb_bands_data, (1, 2, 0)), cmap='gray')

    if original_type=="MAXAR":
        red_band = data[5, :, :]
        green_band = data[3, :, :]
        blue_band = data[2, :, :]
        rgb_composite = np.stack([red_band, green_band, blue_band], axis=-1)
        plt.imshow(rgb_composite)
        plt.axis('off')
        plt.savefig("img.png")

visualize()