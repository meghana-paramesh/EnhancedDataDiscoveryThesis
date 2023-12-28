import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torchvision import transforms
from ImageDataset import NpyDataset, SingleDataset
from torch.utils.data import DataLoader
import joblib

def planet_pca():

    # Step 1: Load and preprocess your dataset
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize to the desired input size
    #     transforms.ToTensor(),
    # ])

    dataset = SingleDataset('/home/CS/mp0157/dataset/planet_tiles_original_new/')

    # dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)  # Replace with your dataset
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Step 2: Apply PCA to reduce the dimensionality
    # Assuming you have 8-band data, you can reduce it to a lower-dimensional space
    n_components = 64  # Adjust as needed
    pca = PCA(n_components=n_components)

    # Collect all data into a single tensor for PCA
    all_data = torch.cat([batch for batch, _ in dataloader], dim=0)
    all_data = all_data.view(all_data.size(0), -1).numpy()  # Flatten each image

    # Fit PCA on your data
    pca.fit(all_data)
    pca_model_filename = 'planet_pca_model.pkl'
    joblib.dump(pca, pca_model_filename)

planet_pca()