# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Transformer(nn.Module):
#     def __init__(self, input_size, output_size, num_layers=2, hidden_size=512, num_heads=8, dropout=0.1):
#         super(Transformer, self).__init__()
#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.embedding(x)
#         for layer in self.transformer_layers:
#             x = layer(x)
#         x = F.relu(self.fc(x))
#         return x

import torch
import torch.nn as nn

class GeospatialTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, hidden_dim=512, num_heads=4, dropout=0.1):
        super(GeospatialTransformer, self).__init__()
        
        # Multi-Head Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads)
        
        # Feedforward Neural Network (MLP) Layer
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            # Multi-Head Self-Attention
            attention_output, _ = self.self_attention(x, x, x)
            x = x + self.dropout(attention_output)
            x = self.norm1(x)
            
            # Feedforward Neural Network
            ff_output = self.feedforward(x)
            x = x + self.dropout(ff_output)
            x = self.norm2(x)
        
        return x