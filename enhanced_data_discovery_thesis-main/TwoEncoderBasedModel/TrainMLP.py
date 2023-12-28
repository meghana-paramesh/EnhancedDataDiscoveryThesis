import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
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
from ReadCorrespondingData import NpyDataset
from Transformer import GeospatialTransformer
from Plot import plot_training_validation_loss, plot_training_loss
from MLPBased import MLP

def train_transformer_mlp():
    print(torch.cuda.is_available())
    device = torch.device("cuda:3")
    transform1, transform2 = get_transform()
    num_epochs = 50
    batch_size = 16

    base_dir = "/home/CS/mp0157/dataset/transformer_train/"
    dataset = NpyDataset(base_dir+"maxar_irrigation_tiles", base_dir+"planet_irrigation_tiles", base_dir+"maxar_urban_tiles", base_dir+"planet_urban_tiles", base_dir+"maxar_snow_cap_mountain_tiles",  base_dir+"planet_snow_cap_mountains_tiles", True, False, 0, 10000, transform1, transform2)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Initialize the image encoders and transformers
    embedding_size = 256

    maxar_model = SimSiam(8, 2048)
    # model= nn.DataParallel(model)
    
    checkpoint_maxar = torch.load('important/maxar_contrastive_loss.pth')
    maxar_model.to(device)
    maxar_model.load_state_dict(checkpoint_maxar['model_state_dict'])

    planet_model = SimSiam(8, 2048) 
    # model= nn.DataParallel(model)
    
    checkpoint_planet = torch.load('important/planet_contrastive_loss.pth')
    planet_model.to(device)
    planet_model.load_state_dict(checkpoint_planet['model_state_dict'])

    # Fine-tune the transformers
    maxar_transformer = MLP(input_dim=embedding_size, hidden_dim=128, output_dim=embedding_size)
    planet_transformer = MLP(input_dim=embedding_size, hidden_dim=128, output_dim=embedding_size)

    maxar_transformer.to(device)
    planet_transformer.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.RAdam(list(maxar_model.parameters()) + list(planet_model.parameters()) +
                        list(maxar_transformer.parameters()) + list(planet_transformer.parameters()), lr=0.0005)

    transformer_train_loss = []
    transformer_validation_loss = []
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_validation_loss = 0.0
        print("epoch: ",epoch)
        for inputs in train_loader:
            maxar1, maxar2, planet1, planet2 = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device)
            z1_online_m, z2_online_m, z1_target_m, z2_target_m = maxar_model(maxar1, maxar2)
            z1_online_p, z2_online_p, z1_target_p, z2_target_p = planet_model(planet1, planet2)

            planet_transformed = maxar_transformer(z1_online_m)
            maxar_transformed = planet_transformer(z1_online_p)

            transformer_loss = loss_fn(planet_transformed, z1_online_p) + loss_fn(maxar_transformed, z1_online_m)
            loss = transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            checkpoint_maxar_transformer = {
                'model_state_dict': maxar_transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            checkpoint_planet_transformer = {
                'model_state_dict': planet_transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint_maxar_transformer, 'important/maxar_MLP.pth')
            torch.save(checkpoint_maxar_transformer, 'important/planet_MLP.pth')
            total_train_loss += loss
            cur_loss = total_train_loss.item()/len(train_loader)
        transformer_train_loss.append(cur_loss)
        print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {cur_loss:.4f}")
    plot_training_loss(transformer_train_loss, "transformer_loss_new", 9)
        

train_transformer_mlp()