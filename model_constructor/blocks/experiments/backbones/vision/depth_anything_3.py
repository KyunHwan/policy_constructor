import sys
import os

from depth_anything_3.api import DepthAnything3
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# Wrapper around https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf

class DepthAnything3Bridge(nn.Module):
    
    def __init__(self, resize_method: str = 'forced'):
        super().__init__()

        self.model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
        self.resize_method = resize_method
        if self.resize_method != 'auto':
            print("Depth Anything v3: Input resolution will be forced to 336 x 504 (h x w)")
    
    def forward(self, image: torch.Tensor, export_feat_layers: list[int]):
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape (B, 3, H, W) on the model device.
                   Need to change it to (B, 1, 3, H, W) internally.
            export_feat_layers: Layer indices to return intermediate features for.

        Returns:
            (Batch, num_features, height, width, feature_dim) shaped latent feature
        """
        assert len(image.shape) == 4 and len(export_feat_layers) != 0
        
        # Scale to [0,1] if it's likely 0–255
        if image.max().item() > 1.5:
            image = image / 255.0

        def nearest_multiple(x: int, m: int) -> int:
            # nearest integer multiple of m (ties go up)
            return max(m, ((x + (m // 2)) // m) * m)
        
        # DA3 requires H,W to be multiples of 14
        if self.resize_method == 'auto':
            h = nearest_multiple(image.shape[-2], 14)
            w = nearest_multiple(image.shape[-1], 14)
    
            if (h, w) != image.shape[-2:]:
                image = F.interpolate(
                    image,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            image = F.interpolate(
                        image,
                        size=(336, 504),
                        mode="bilinear",
                        align_corners=False,
                    )
        latent_features = None
        latent_list = []
        latent_feature_output_dict = self.model(image=einops.rearrange(image, 'b c h w -> b 1 c h w'), 
                            export_feat_layers=export_feat_layers)['aux']
        for key in latent_feature_output_dict.keys():
            latent_list.append(latent_feature_output_dict[key])
        latent_features = torch.cat(latent_list, dim=1)
            
        return latent_features


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(device)
    print(f"Total parameters of model: {sum(p.numel() for p in model.parameters())}")