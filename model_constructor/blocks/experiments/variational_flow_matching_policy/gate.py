import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import math

class Gate(nn.Module):
    def __init__(self, input_dim, num_experts, temperature, top_k):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k

        self.model = nn.Sequential(
            *[
                nn.Linear(self.input_dim, self.input_dim * 2),
                nn.LayerNorm(self.input_dim * 2),
                nn.ELU(),
                nn.Linear(self.input_dim * 2, self.input_dim * 2),
                nn.LayerNorm(self.input_dim * 2),
                nn.ELU(),
                nn.Linear(self.input_dim * 2, self.num_experts),
                nn.LayerNorm(self.num_experts),
            ]
        )
        
    def forward(self, input: torch.Tensor, iterations: int, training: bool=False, use_noise: bool=False,):
        """
        Args:
            input: (batch, feature_dim) shape
        
        Return:
            (batch, num_experts) shape
        """

        if len(input.shape) == 3:
            input = einops.rearrange(input, 'b 1 d -> b d')

        clean_logits = self.model(input)

        if training and use_noise:
            noisy_logits = clean_logits + torch.randn_like(clean_logits)
        else:
            noisy_logits = clean_logits

        router_probs = F.softmax(noisy_logits / self.temperature, dim=-1)

        return router_probs
