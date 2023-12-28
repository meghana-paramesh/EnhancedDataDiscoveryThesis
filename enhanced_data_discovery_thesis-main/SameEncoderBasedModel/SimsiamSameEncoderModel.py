import torch
import torch.nn as nn
import torchvision.models as models



class SimSiam(nn.Module):
    def __init__(self, projection_dim=2048):
        super(SimSiam, self).__init__()
        num_output_layers=3

        self.encoder = models.resnet50(weights=False)
        self.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify the first convolutional layer to accept 8 channels
        self.encoder.fc = nn.Identity()  # Remove the classifier layer

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

        # self.projector = nn.Sequential(
        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048)
        # )

        # self.classifier = nn.Linear(2048, num_output_layers)

    def forward(self, x1, x2):
        # z1 = self.predictor(self.projector(self.encoder(x1)))
        # z2 = self.predictor(self.projector(self.encoder(x2)))
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2



# Initialize the SimSiam model
# input_dim = 8
# model = SimSiam(input_dim)
#
# # Generate two random input images with batch dimension
# x1 = torch.randn(64, input_dim, 256, 256)
# x2 = torch.randn(64, input_dim, 256, 256)

# Forward pass through the model
# z1, p1, z2, p2 = model(x1, x2)

# # Print the output shapes
# print("z1 shape:", z1.shape)
# print("p1 shape:", p1.shape)
# print("z2 shape:", z2.shape)
# print("p2 shape:", p2.shape)

