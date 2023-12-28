import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_heads=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.layer_norm(x)
        
        return x