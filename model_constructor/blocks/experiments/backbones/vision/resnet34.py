import torchvision.models as models
import torch.nn as nn
import einops
import torch.nn.functional as F

class Resnet34(nn.Module):
    def __init__(self, resize: bool, resize_spec: list[int, int] | None):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.image_resize = resize_spec
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
            
        if self.resize and self.image_resize is not None:
            h, w = self.image_resize
            image = F.interpolate(
                        image,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
        
        return self.model(image)
