import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import einops
import torch.nn.functional as F

class Resnet34Group(nn.Module):
    def __init__(self, resize: bool=True):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'head': nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2]),
            'left': nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2]),
            'right': nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2]),
        })

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.resize = resize
    
    def forward(self, images: dict[str, torch.Tensor]):
        """
        Args:
            images:
                -head: (B, C, H, W)
                -left: (B, C, H, W)
                -right: (B, C, H, W)
        """
        for key in images.keys():
            if images[key].max().item() > 1.5:
                images[key] = images[key] / 255.0
            if len(images[key].shape) == 3:
                images[key] = einops.rearrange(images[key], 'c h w -> 1 c h w')
            if self.resize:
                h, w = (240, 320)
                images[key] = F.interpolate(
                            images[key],
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )
            images[key] = self.normalize(images[key])
        
        returned_dict = {}
        for key in images.keys():
            returned_dict[key] = self.encoders[key](images[key])

        return returned_dict
