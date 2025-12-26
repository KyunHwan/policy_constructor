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
                memory_input: torch.Tensor, # memory input as cross-attention
                discrete_semantic_input: torch.Tensor | None=None,
                **kwargs) -> torch.Tensor: # other latent features (ex. language)
        raise NotImplementedError
