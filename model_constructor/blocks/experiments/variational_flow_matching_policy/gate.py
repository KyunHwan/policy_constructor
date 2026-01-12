import torch
import torch.nn as nn
import einops

class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts

        self.model = nn.Sequential(
            *[
                nn.Linear(self.input_dim, self.input_dim * 2),
                nn.LayerNorm(self.input_dim * 2),
                nn.ELU(),
                nn.Linear(self.input_dim * 2, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ELU(),
                nn.Linear(self.input_dim, self.num_experts),
                nn.LayerNorm(self.num_experts),
            ]
        )
        self.last_gating_func = nn.Softmax(dim=1)
        
    def forward(self, input):
        """
        Args:
            input: (batch, feature_dim) shape
        
        Return:
            (batch, num_experts) shape
        """
        if len(input.shape) == 3:
            input = einops.rearrange(input, 'b 1 d -> b d')
        input = nn.functional.normalize(input, dim=1)
        return self.last_gating_func(self.model(input))

