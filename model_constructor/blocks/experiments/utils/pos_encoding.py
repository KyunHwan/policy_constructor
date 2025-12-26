import torch
import math

def get_sinusoidal_pos_encoding(seq_len, d_model, device):
    """
    Standard sinusoidal positional embeddings (not learnable).
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0) # [1, Seq_Len, D_Model]