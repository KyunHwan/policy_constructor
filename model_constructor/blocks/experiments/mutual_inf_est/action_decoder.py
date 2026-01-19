import torch
import torch.nn as nn
import einops

class ActionDecoder(nn.Module):
    def __init__(self,
                 final_activation="sigmoid",
                 embedding_dim: int=24,
                 hidden_dim: int=1024,
                 action_dim: int=24,
                 action_chunk: int=40):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk

        self.decoder = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * action_chunk),
            nn.ReLU(inplace=True),
        ])

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, dim)
        """
        
        return {
            'action': self.decoder(embedding).view(embedding.shape[0], self.action_chunk, self.action_dim)
        }