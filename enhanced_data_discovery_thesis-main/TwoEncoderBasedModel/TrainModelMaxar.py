import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import torch.optim as optim
from torchvision import models
import torch

from ImageDataset import NpyDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import faiss
import numpy as np
import os
from natsort import natsorted
from Plot import plot_training_validation_loss, plot_training_loss
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from Transforms import get_transform
from ContrastiveLoss import ContrastiveLoss


def train():
   # TODO: test the model
   # Initialize the SimSiam model
   device = torch.device("cuda:0")
   transform1, transform2 = get_transform()

   
   num_epochs = 50
   batch_size = 32

   base_dir = "/home/CS/mp0157/dataset/new_tiles_original/"

   dataset_maxar = NpyDataset(base_dir+"maxar_irrigation_tiles", base_dir+"maxar_urban_tiles", base_dir+"maxar_snow_cap_mountain_tiles", True, False, 0, 20000, transform1, transform2)

   train_size = int(0.9 * len(dataset_maxar))
   validation_size = len(dataset_maxar) - train_size
   train_dataset, validation_dataset = torch.utils.data.random_split(dataset_maxar, [train_size, validation_size])

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


   training_loss = []
   validation_loss = []
   model = SimSiam(8, 2048)
   # model= nn.DataParallel(model)
   model.to(device)
   optimizer = optim.RAdam(model.parameters(), lr=0.001)
   # optimizer = optimizer.to(device)


   criterion = ContrastiveLoss(temperature=0.5)

   criterion = criterion.to(device)
   print(len(train_dataset))

   for epoch in range(num_epochs):
      print("Started: ", epoch)
      model.train()
      total_loss = 0.0

      for inputs in train_loader:
         
         x1, x2 = inputs[0].to(device), inputs[1].to(device)

         optimizer.zero_grad()
         z1_online, z2_online, z1_target, z2_target = model(x1, x2)
         loss = criterion(z1_online, z2_online, z1_target, z2_target)
         loss = loss.mean()
         loss.backward()
         optimizer.step()

         total_loss += loss.item()

      # Calculate training loss
      train_loss = total_loss / len(train_loader)
      training_loss.append(train_loss)

      # Validation step
      model.eval()  # Set the model to evaluation mode
      val_loss = 0.0
      with torch.no_grad():
         for val_inputs in validation_loader:
               x1_val, x2_val = val_inputs[0].to(device), val_inputs[1].to(device)
               p1_val, p2_val, z1_val, z2_val = model(x1_val, x2_val)
               loss_val = criterion(p1_val, p2_val, z1_val, z2_val)
               loss_val = loss_val.mean()
               val_loss += loss_val.item()

      val_loss /= len(validation_loader)
      validation_loss.append(val_loss)
      checkpoint = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
      }

      torch.save(checkpoint, 'maxar_contrastive_loss1.pth')
      print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}")
   
   plot_training_validation_loss(training_loss, validation_loss, "maxar_contrastive_loss1", 5)

train()



