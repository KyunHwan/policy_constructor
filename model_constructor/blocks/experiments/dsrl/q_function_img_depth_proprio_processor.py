import torch
import torch.nn as nn
import einops

class NoiseActorImgDepthProprioProcessor(nn.Module):
    def __init__(self,
                 depth_data_keys: list = ['head_depth'],
                 img_data_keys: list = ['head', 'left', 'right'],
                 num_latent_depth_feats: int=1,
                 num_latent_img_feats: int=3,
                 input_img_channel: int=64,
                 input_depth_channel: int=64,
                 output_img_channel: int=24,
                 output_depth_channel: int=24
                ):
        super().__init__()
        self.depth_data_keys = depth_data_keys
        self.img_data_keys = img_data_keys
        self.num_latent_depth_feats = num_latent_depth_feats
        self.num_latent_img_feats = num_latent_img_feats
        self.input_img_channel = input_img_channel
        self.input_depth_channel = input_depth_channel
        self.output_img_channel = output_img_channel
        self.output_depth_channel = output_depth_channel

        self.img_latent_vec_proj = nn.ModuleDict({
            key: nn.Conv2d(self.input_img_channel, self.output_img_channel, kernel_size=1) for key in self.img_data_keys
        })
        self.depth_latent_vec_proj = nn.ModuleDict({
            key: nn.Conv2d(self.input_depth_channel, self.output_depth_channel, kernel_size=1) for key in self.depth_data_keys
        })

    def forward(self, data: dict[str, torch.Tensor]):
        """
        data:
            - head_depth: (batch, num_channels, height, width)
            - head: (batch, num_channels, height, width)
            - left: (batch, num_channels, height, width)
            - right: (batch, num_channels, height, width)
            - proprio: (batch, obs_history, obs_dim)
            - action: (batch, action_chunk_size, action_dim)
        """

        flattened_depth = None
        for key in self.depth_data_keys:
            if flattened_depth is None:
                flattened_depth = einops.rearrange(self.depth_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')
            else:
                flattened_depth = torch.concat((flattened_depth, einops.rearrange(self.depth_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')), dim=1)
        
        flattened_img = None
        for key in self.img_data_keys:
            if flattened_img is None:
                flattened_img = einops.rearrange(self.img_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')
            else:
                flattened_img = torch.concat((flattened_img, einops.rearrange(self.img_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')), dim=1)
        
        return torch.concat((
                flattened_depth,
                flattened_img,
                einops.rearrange(data['proprio'], 'b h d -> b (h d)'),
                einops.rearrange(data['action'], 'b c d -> b (c d)'),
                ), dim=1)
