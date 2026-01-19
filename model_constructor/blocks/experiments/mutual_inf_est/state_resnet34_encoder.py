from ..backbones.vision.resnet34 import Resnet34

import torch
import torch.nn as nn
import einops

class ResNet34EncoderGroup(nn.Module):
    def __init__(self, 
                 num_images: int=3, 
                 img_embedding_dim: int=24,
                 state_embedding_dim: int=24,
                 hidden_dim: int=1024,
                 state_dim: int=24,
                 state_chunk: int=40,
                 img_size: int=128,
                 input_channels: int=512,):
        super().__init__()
        self.img_embedding_dim = img_embedding_dim
        self.input_channels = input_channels
        self.img_dec_img_size = int(img_size/32)
        self.num_images = num_images
        
        self.img_encoders = nn.ModuleList([
            nn.Sequential(*[
                Resnet34(resize=False, resize_spec=[img_size, img_size]),
                nn.Softmax2d()
            ])
            for _ in range(num_images)
        ])

        # self.img_emb_to_emb = nn.ModuleList([ # (B, input_channels, img_dec_img_size, img_dec_img_size) --> (B, input_channels, img_input_size/32, img_input_size/32)(B, embedding_dim) -->  --> 
        #     nn.Sequential(*[
        #         nn.Flatten()
        #         nn.Linear(input_channels * self.img_dec_img_size * self.img_dec_img_size, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, self.img_embedding_dim),
        #         nn.ReLU(inplace=True),
        #     ])
        #     for _ in range(num_images)
        # ])

        self.state_encoder = nn.Sequential(*[
                nn.Linear(num_images * input_channels * self.img_dec_img_size * self.img_dec_img_size + state_dim * state_chunk, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.img_embedding_dim * num_images + state_embedding_dim),
                nn.ReLU(inplace=True),
            ])


    def forward(self, data: dict[str, torch.Tensor]):
        """
        Args:
            data:

        """
        assert len(data.keys()) == self.num_images + 1

        outputs = []
        for i, key in enumerate(data.keys()):
            if key != 'state':
                outputs.append(nn.Flatten(self.img_encoders[i](data[key])))
            else:
                outputs.append(nn.Flatten(data[key]))
        return {
            'embedding': self.state_encoder(torch.cat(outputs, dim=1))
        }