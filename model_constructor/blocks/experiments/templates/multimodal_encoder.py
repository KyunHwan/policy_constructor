from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class MultiModalEncoderTemplate(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self,
                cond_proprio: torch.Tensor, # latent proprio features
                cond_visual: torch.Tensor, # latent visual features
                cond_semantic: torch.Tensor | None=None, # latent semantic features
                action: torch.Tensor | None=None, # latent action features
                **kwargs): # other latent features (ex. language)
        raise NotImplementedError

