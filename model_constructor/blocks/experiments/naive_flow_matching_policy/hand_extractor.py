from ..backbones.vision.resnet34 import Resnet34

import torch
import torch.nn as nn
import einops

class ResNet34Encoder(nn.Module):
    def __init__(self, 
                 img_h, 
                 img_w, 
                 img_resize, 
                 self_extracted_dim,
                 output_dim):
        super().__init__()
        self.encoder = Resnet34(resize=img_resize, resize_spec=[img_h, img_w])
        self.self_extracted_dim = self_extracted_dim
        self.output_dim = output_dim
        self.feature_projector = None
        if self.self_extracted_dim != self.output_dim:
            self.feature_projector = nn.Sequential(
                *[
                    nn.LayerNorm(self_extracted_dim),
                    nn.Linear(self_extracted_dim, self.output_dim),
                ]
            )

    def forward(self, img):
        if self.feature_projector is not None:
            return self.feature_projector(einops.rearrange(self.encoder(img), 'b c h w -> b (h w) c'))
        else:
            return einops.rearrange(self.encoder(img), 'b c h w -> b (h w) c')
