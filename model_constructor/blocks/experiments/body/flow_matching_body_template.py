from abc import ABC, abstractmethod
import torch
import torch.nn

class FlowMatchingBodyTemplate(ABC, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, 
                time: float,
                noise: torch.Tensor, # noise input to the model
                cond_proprio: torch.Tensor, # latent proprio features
                cond_visual: torch.Tensor, # latent visual features
                **kwargs) -> torch.Tensor: # other latent features (ex. language)
        raise NotImplementedError
