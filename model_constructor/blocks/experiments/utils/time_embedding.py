import math
import torch
import torch.nn as nn

def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000):
    """
    Standard sinusoidal time embedding.
    :param timesteps: (batch, ) or (batch, 1) tensor of values [0, 1] or steps
    :param embedding_dim: dimension of the output embedding
    """
    assert embedding_dim % 2 == 0, "Embedding dim must be even for sinusoidal encoding"
    
    half_dim = embedding_dim // 2
    # Logic: exp(log(max_pos) * i / half_dim)
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    
    # timesteps * frequencies
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    return emb