import torch
import torch.nn as nn
import einops

class ActionEncoder(nn.Module):
    def __init__(self,
                 final_activation="sigmoid",
                 embedding_dim: int=24,
                 hidden_dim: int=1024,
                 action_dim: int=24,
                 action_chunk: int=40):
        super().__init__()
        self.encoder = nn.Sequential(*[
            nn.Linear(action_dim * action_chunk, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
        ])

    def forward(self, action):
        """
        Args:
            action: (batch, action chunk, action dim)
        """
        if len(action.shape) == 3:
            action = einops.rearrange(action, 'b c d -> b (c d)')
        return {
            'embedding': self.encoder(action)
        }