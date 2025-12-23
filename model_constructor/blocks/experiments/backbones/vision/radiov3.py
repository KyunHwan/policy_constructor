import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn 

class RadioV3(nn.Module):
    def __init__(self, channels=(1024, 3072), device="cuda"):
        super().__init__()
        self._channels = channels # (img features, img summary features)
        self._device = torch.device(device)

        # Load C-RADIOv3-L from TorchHub
        self.radiov3_version = "c-radio_v3-l" # 448 x 448 (height x width) --> (28, 28) w/ channel size 1024
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=self.radiov3_version,
            progress=True,
        ).to(self.device).eval()

    @property
    def num_channels(self):
        return self._channels

    @property
    def device(self):
        return self._device

    def forward(self, x: torch.Tensor):
        """
        x: (C,H,W) or (B,C,H,W), values in [0,255] or [0,1].
        Returns:
            ([features], [summary])
                summary: (B, C_summary)
                features: (B, C_feat, H_feat, W_feat)  if feature_fmt='NCHW'
        """
        # Ensure batch dimension
        if x.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        # Move to device and float32
        if not x.is_cuda:
            x = x.to(self.device, dtype=torch.float32)

        # Scale to [0,1] if it's likely 0–255
        if x.max().item() > 1.5:
            x = x / 255.0

        # RADIO requires H,W to be multiples of min_resolution_step
        nearest_res = self.model.get_nearest_supported_resolution(
            x.shape[-2],
            x.shape[-1],
        )
        h, w = nearest_res.height, nearest_res.width
        if (h, w) != x.shape[-2:]:
            x = F.interpolate(
                x,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            # Ask for NCHW feature format so we get a conv-like feature map
            summary, features = self.model(x, feature_fmt="NCHW")

        # Wrap in lists to match BaseBackbone interface (single scale)
        return features, summary


def run_radiov3_test(image_path: str, target_size=(640, 512), device: str = "cpu"):
    from PIL import Image
    """
    Utility for quickly trying RadioV3 inference on a single image.
    It resizes, shows the image, then prints the shapes of the returned tensors.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    radio = RadioV3(device=device)
    radio.eval()

    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize(target_size, Image.BILINEAR)
    resized_image.show(title="RadioV3 resized input")

    # CLIPImageProcessor accepts torch tensors, so keep channel-first.
    image_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float()

    with torch.no_grad():
        feature_list, summary_list = radio(image_tensor)

    print(f"Resized image size (width x height): {resized_image.size}")
    print(f"Feature tensor shape: {feature_list[0].shape}")
    print(f"Summary tensor shape: {summary_list[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick RadioV3 single-image inference test.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--width", type=int, default=320, help="Resize width for the input image.")
    parser.add_argument("--height", type=int, default=256, help="Resize height for the input image.")
    parser.add_argument("--device", default="cpu", help="torch device to run RadioV3 on (e.g., cpu or cuda).")
    args = parser.parse_args()

    run_radiov3_test(args.image, target_size=(args.width, args.height), device=args.device)