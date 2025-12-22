import torch
import torch.nn as nn

class DUNE(nn.Module):
    """
    https://europe.naverlabs.com/research/publications/dune/
    """
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("naver/dune", "dune_vitbase_14_448_paper")
    
    def forward(self, image: torch.Tensor):
        # need to check for size (448 x 448)

        return self.model(image)