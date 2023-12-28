import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# SimSiam Architecture
class SimSiam(nn.Module):
    def __init__(self, bands, base_encoder, projection_dim=256):
        super(SimSiam, self).__init__()
        num_classes=3
        self.encoder = models.resnet50(weights=False)
        self.encoder.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify the first convolutional layer to accept 8 channels
        self.encoder.fc = nn.Identity()

        self.online_encoder = nn.Sequential(
            self.encoder,
            nn.BatchNorm1d(2048),  # Adjust the number of features to match the base encoder's output
            nn.Linear(2048, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Linear(projection_dim, projection_dim),
        )
        self.target_encoder = nn.Sequential(
            self.encoder,
            nn.BatchNorm1d(2048),  # Adjust the number of features to match the base encoder's output
            nn.Linear(2048, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x1, x2):
        z1_online, z2_online = self.online_encoder(x1), self.online_encoder(x2)
        z1_target, z2_target = self.target_encoder(x1), self.target_encoder(x2)
        return z1_online, z2_online, z1_target, z2_target