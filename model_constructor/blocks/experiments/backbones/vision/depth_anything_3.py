import torch
import torch.nn as nn
from .depth_anything_3.api import DepthAnything3


class DA3(nn.Module):
    """
    https://github.com/ByteDance-Seed/depth-anything-3
    """
    def __init__(self):
        super().__init__()
        self.model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
    
    def forward(self, image: torch.Tensor):
        """
        Arguments:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.

        Returns:
            Dictionary containing model predictions

            print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
            print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
            print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
            print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
        """
        return self.model(image)
