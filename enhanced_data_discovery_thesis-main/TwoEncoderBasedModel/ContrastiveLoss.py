import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1_online, z2_online, z1_target, z2_target):
        online_targets = F.normalize(z2_target.detach(), dim=-1)
        target_targets = F.normalize(z1_target.detach(), dim=-1)
        
        loss = -(F.cosine_similarity(z1_online, online_targets, dim=-1) / self.temperature).exp().mean() - \
               (F.cosine_similarity(z2_online, target_targets, dim=-1) / self.temperature).exp().mean()
        
        return loss