import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.ops import MLP
import torchvision.models as models
import torchvision.transforms as transforms

import einops

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


class ResidualActorPreprocessor(nn.Module):
    def __init__(self,
                 action_key = 'action',
                 proprio_key = 'proprio',
                 depth_data_keys: list = ['head_depth'],
                 img_data_keys: list = ['head', 'left', 'right'],
                 input_img_channel: int=64,
                 input_depth_channel: int=64,
                 output_img_channel: int=24,
                 output_depth_channel: int=24
                ):
        super().__init__()
        self.action_key = action_key
        self.proprio_key = proprio_key
        self.depth_data_keys = depth_data_keys
        self.img_data_keys = img_data_keys
        self.num_latent_depth_feats = len(depth_data_keys)
        self.num_latent_img_feats = len(img_data_keys)
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
            - action: (batch, 1, action_dim)
        """

        flattened_output = None
        flattened_depth = None
        for key in self.depth_data_keys:
            if flattened_depth is None:
                flattened_depth = einops.rearrange(self.depth_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')
            else:
                flattened_depth = torch.concat((flattened_depth, einops.rearrange(self.depth_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')), dim=1)
        
        if flattened_depth is not None:
            flattened_output = flattened_depth

        flattened_img = None
        for key in self.img_data_keys:
            if flattened_img is None:
                flattened_img = einops.rearrange(self.img_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')
            else:
                flattened_img = torch.concat((flattened_img, einops.rearrange(self.img_latent_vec_proj[key](data[key]), 'b c h w -> b (c h w)')), dim=1)

        if flattened_output is None:
            flattened_output = flattened_img
        else:
            if flattened_img is not None:
                flattened_output = torch.concat((flattened_output, flattened_img), dim=1)

        return torch.concat((
                flattened_output,
                einops.rearrange(data[self.proprio_key], 'b h d -> b (h d)'),
                einops.rearrange(data[self.action_key], 'b c d -> b (c d)'),
                ), dim=1)


class Residual_Actor(nn.Module):
    def __init__(self, 
                 img_resize,

                 action_key = 'action',
                 proprio_key = 'proprio',
                 depth_data_keys: list = ['head_depth'],
                 img_data_keys: list = ['head', 'left', 'right'],
                 input_img_channel: int=64,
                 input_depth_channel: int=64,
                 output_img_channel: int=24,
                 output_depth_channel: int=24,

                 input_dim=64,
                 action_dim: int=32, # 40 actions with 24 action dimension
                 num_layers: int=3,
                 num_hidden_dim: int=2048,
                 dropout: float=0.0,
                 init_method = 'xavier'
                 ):
        super().__init__()

        self.resnet_group = Resnet34Group(resize=img_resize)
        self.preprocessor = ResidualActorPreprocessor(
            action_key = action_key,
            proprio_key = proprio_key,
            depth_data_keys = depth_data_keys,
            img_data_keys = img_data_keys,
            input_img_channel = input_img_channel,
            input_depth_channel = input_depth_channel,
            output_img_channel = output_img_channel,
            output_depth_channel = output_depth_channel
        )

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.init_method = init_method

        self.model = MLP(
            in_channels = self.input_dim,
            hidden_channels = [*[num_hidden_dim for _ in range(num_layers)], num_hidden_dim],
            activation_layer = nn.ELU,
            norm_layer = nn.LayerNorm,
            dropout = dropout
        )
        self.last_layer = nn.Sequential(
                nn.Linear(
                    in_features = num_hidden_dim, 
                    out_features = self.action_dim
                ),
                nn.Tanh() # in order to prevent going outside the domain of the Gaussian noise
        )

        # === apply init AFTER modules are built ===
        if self.init_method == "xavier":
            self.model.apply(self._xavier_init_fn)
            self.last_layer.apply(self._xavier_init_fn)

    def forward(self, state, action):
        """
        data:
            - head_depth: (batch, num_channels, height, width)
            - head: (batch, num_channels, height, width)
            - left: (batch, num_channels, height, width)
            - right: (batch, num_channels, height, width)
            - proprio: (batch, obs_history, obs_dim)
            - action: (batch, 1, action_dim)

        action: (batch, action_dim)

        outputs: (batch, action_dim)
        """
        imgs = {
            'head': state['head'],
            'left': state['left'],
            'right': state['right']
        }

        # 2. Pass them through the ResNet group
        processed_imgs = self.resnet_group(imgs)

        # 3. Build preprocessor input without mutating the caller's state dict
        preprocessor_input = {
            **processed_imgs,
            'proprio': state['proprio'],
            'action': action,
        }

        # 4. Pass to the preprocessor
        flattened_tensor = self.preprocessor(preprocessor_input)

        intermediate_output = self.model(flattened_tensor)
        updated_action = self.last_layer(intermediate_output)
        
        return updated_action

    def _xavier_init_fn(self, m: nn.Module):
        """
        This runs on every submodule when called via .apply(...)
        We only touch Linear and Conv weights.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict, strict=True)
    