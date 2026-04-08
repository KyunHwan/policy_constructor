import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.ops import MLP
import torchvision.models as models
import torchvision.transforms as transforms

import random

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


class QFunctionPreprocessor(nn.Module):
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


class Q_Function(nn.Module):
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
                 num_ensemble: int=3,
                 num_layers: int=3,
                 num_hidden_dim: int=2048,
                 dropout: float=0.0,
                 init_method = 'xavier'
                 ):
        """
        Each Q function is an MLP with num_layers and num_hidden_dim with activation and normalization scheme.
        For example, activation can be tanh or GELU and normalization can be LayerNorm or BatchNorm.
        """
        super().__init__()

        self.resnet_group = Resnet34Group(resize=img_resize)
        self.preprocessor = QFunctionPreprocessor(
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
        self.init_method = init_method
        self.num_ensemble = num_ensemble
        
        self.num_hidden_dim = num_hidden_dim
        self.q_ensembles_hidden_layers = nn.ModuleList(
            [
                MLP(
                    in_channels = self.input_dim,
                    hidden_channels = [*[self.num_hidden_dim for _ in range(num_layers)], self.num_hidden_dim],
                    activation_layer = nn.ELU,
                    norm_layer = nn.LayerNorm,
                    dropout = dropout
                )
                for _ in range(self.num_ensemble)
            ]
        )
        self.q_ensembles_output_layers = nn.ModuleList(
            [nn.Linear(
                    in_features = self.num_hidden_dim, 
                    out_features = 1
                )
                for _ in range(self.num_ensemble)
            ]
        )

        # === apply init AFTER modules are built ===
        if self.init_method == "xavier":
            # don't touch vision_backend (likely pretrained)
            self.q_ensembles_hidden_layers.apply(self._xavier_init_fn)
            self.q_ensembles_output_layers.apply(self._xavier_init_fn)
    
    def forward(self, data, subsample_q=True, critic=False):#flattened_tensor):
        """
        flattened_tensor: (batch, dim)

        Outputs mean value of Q function ensemble
        (batch, mean_val)

        data:
            - head_depth: (batch, num_channels, height, width)
            - head: (batch, num_channels, height, width)
            - left: (batch, num_channels, height, width)
            - right: (batch, num_channels, height, width)
            - proprio: (batch, obs_history, obs_dim)
            - action: (batch, action_chunk_size, action_dim)
        """
        imgs = {
            'head': data['head'],
            'left': data['left'],
            'right': data['right']
        }

        # 2. Pass them through the ResNet group
        processed_imgs = self.resnet_group(imgs)

        # 3. Build preprocessor input without mutating the caller's data dict
        preprocessor_input = {
            **processed_imgs,
            'proprio': data['proprio'],
            'action': data['action'],
        }

        # 4. Pass to the preprocessor
        flattened_tensor = self.preprocessor(preprocessor_input)

        
        if not subsample_q:
            if not critic:
                running_sum = 0.0
                for i in range(self.num_ensemble):
                    Q_i_hidden_layer_output = self.q_ensembles_hidden_layers[i](flattened_tensor)
                    Q_i_val = self.q_ensembles_output_layers[i](Q_i_hidden_layer_output)
                    running_sum += Q_i_val
                
                return running_sum / self.num_ensemble
            else:
                indices = random.sample(range(self.num_ensemble), k=max(self.num_ensemble // 2, 1))
                running_sum = 0.0
                for i in indices:
                    Q_i_hidden_layer_output = self.q_ensembles_hidden_layers[i](flattened_tensor)
                    Q_i_val = self.q_ensembles_output_layers[i](Q_i_hidden_layer_output)
                    running_sum += Q_i_val
                
                return running_sum / max(self.num_ensemble // 2, 1)

        
        else:
            q_values = []
            indices = random.sample(range(self.num_ensemble), k=max(self.num_ensemble // 2, 1))
            for i in indices:
                Q_i_hidden_layer_output = self.q_ensembles_hidden_layers[i](flattened_tensor)
                Q_i_val = self.q_ensembles_output_layers[i](Q_i_hidden_layer_output)
                q_values.append(Q_i_val)

            return torch.min(torch.stack(q_values), dim=0).values
    
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
