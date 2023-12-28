import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def custom_color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    num_bands = img.shape[0]
    for band in range(num_bands):
        img[band] = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )(img[band])
    return img

def get_transform():

    transform_view1 = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), ),
    ])

    transform_view2 = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), ),
    ])

    return transform_view1, transform_view2