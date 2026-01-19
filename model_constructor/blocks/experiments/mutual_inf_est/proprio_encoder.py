import torch
import torch.nn as nn

class ProprioEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 num_layers: int, 
                 hidden_dim: int,
                 output_dim: int):
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, input):
        
