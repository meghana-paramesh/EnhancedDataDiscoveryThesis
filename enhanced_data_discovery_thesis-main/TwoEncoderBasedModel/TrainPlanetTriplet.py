import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from TripletDataset import NpyDataset
from Plot import plot_training_validation_loss, plot_training_loss

# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)
        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0).mean()
        return loss

# Define the triplet network architecture
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(256 * 256 * 8, 128),  # Adjust input size based on your image dimensions
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, x):
        return self.embedding(x)

# Create DataLoader for the triplet dataset
device1 = torch.device("cuda:3")

num_epochs = 50
batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])
device1 = torch.device("cuda:2")

base_dir = "/home/CS/mp0157/dataset/new_tiles_original/"
dataset_planet = NpyDataset(base_dir+"planet_irrigation_tiles", base_dir+"planet_urban_tiles", base_dir+"planet_snow_cap_mountains_tiles",True, False, transform)
triplet_dataloader = DataLoader(dataset_planet, batch_size=32, shuffle=True)

# Initialize the triplet network and optimizer
triplet_net = TripletNet()
triplet_net.to(device1)
triplet_loss_fn = TripletLoss(margin=0.2)
optimizer = optim.Adam(triplet_net.parameters(), lr=0.001)

# Training loop
num_epochs = 50
training_loss = []

for epoch in range(num_epochs):
    triplet_net.train()
    total_loss = 0.0

    for anchor, positive, negative in triplet_dataloader:
        anchor, positive, negative = anchor.view(anchor.size(0), -1).to(device1), positive.view(positive.size(0), -1).to(device1), negative.view(negative.size(0), -1).to(device1)
        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = triplet_net(anchor), triplet_net(positive), triplet_net(negative)
        loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    cur_loss = total_loss/len(triplet_dataloader)
    training_loss.append(cur_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(triplet_dataloader)}")
    checkpoint = {
      'model_state_dict': triplet_net.state_dict(),
      'optimizer_state_dict': triplet_net.state_dict()
      }

    torch.save(checkpoint, 'planet_triplet.pth')
    print(f"Epoch [{epoch}/{num_epochs}] Planet Training Loss: {cur_loss}")
plot_training_loss(training_loss, "planet_triplet", 1)


# After training, you can use the trained triplet_net to obtain embeddings for your images