import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

import einops
import torch.nn.functional as F

class Resnet34(nn.Module):
    def __init__(self, resize: bool=True):
        super().__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.model = nn.Sequential(*list(self.encoder.children())[:-2])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.resize = resize
    
    def forward(self, image):
        """
        Args:
            image: needs to be (B, C, H, W)
        """
        if image.max().item() > 1.5:
            image = image / 255.0

        if len(image.shape) == 3:
            image = einops.rearrange(image, 'c h w -> 1 c h w')
            
        if self.resize:
            h, w = (240, 320)
            image = F.interpolate(
                        image,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
            
        image = self.normalize(image)

        return self.model(image)
