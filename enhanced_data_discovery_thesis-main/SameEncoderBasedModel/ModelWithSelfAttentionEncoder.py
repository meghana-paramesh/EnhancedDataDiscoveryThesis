import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define a simple self-attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads

        # Reshape the input channels to match the required format for nn.MultiheadAttention
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Multi-head attention module
        self.multihead_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)

        # Output projection
        self.output_projection = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        q = self.query(x).view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        k = self.key(x).view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        v = self.value(x).view(x.size(0), x.size(1), -1).permute(2, 0, 1)

        # Compute self-attention using MultiheadAttention
        attn_output, _ = self.multihead_attention(q, k, v)

        # Reshape and project back to original size
        attn_output = attn_output.permute(1, 2, 0).contiguous().view(x.size())
        attn_out = self.output_projection(attn_output)

        return attn_out

# Define the SimSiam model
class SimSiam(nn.Module):
    def __init__(self, input_channels, feature_dim=2048, num_heads=8):
        super(SimSiam, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SelfAttentionBlock(64, 64, num_heads=num_heads),  # Self-Attention Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SelfAttentionBlock(128, 128, num_heads=num_heads),  # Self-Attention Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SelfAttentionBlock(256, 256, num_heads=num_heads),  # Self-Attention Block
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(256 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # x1 and x2 are two augmented views of the same input
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)

        proj1 = self.projector(feat1.flatten(start_dim=1))
        proj2 = self.projector(feat2.flatten(start_dim=1))

        pred1 = self.predictor(proj1)
        pred2 = self.predictor(proj2)

        return proj1, proj2, pred1, pred2