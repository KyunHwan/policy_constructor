import torch
import torch.nn as nn
import einops

class ResNet34DecoderGroup(nn.Module):
    def __init__(self, 
                 out_channels: int=3,
                 final_activation: str="sigmoid", 
                 num_images: int=3, 
                 img_embedding_dim: int=24,
                 state_embedding_dim: int=24,
                 hidden_dim: int=1024,
                 state_dim: int=24,
                 state_chunk: int=40,
                 img_size: int=128,
                 input_channels: int=512,):
        super().__init__()
        self.embedding_dim = img_embedding_dim * num_images + state_embedding_dim
        self.input_channels = input_channels
        self.img_dec_img_size = int(img_size/32)
        
        self.emb_to_img_dec = nn.ModuleList([ # (B, embedding_dim) --> (B, input_channels, img_input_size/32, img_input_size/32) --> (B, out_channels, img_input_size, img_input_size)
            nn.Sequential(*[
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, input_channels * self.img_dec_img_size * self.img_dec_img_size),
                nn.ReLU(inplace=True),
            ])
            for _ in range(num_images)
        ])
        self.img_decoders = nn.ModuleList([
            ResNet34Decoder(out_channels=out_channels, norm_layer=nn.BatchNorm2d, final_activation="sigmoid") 
            for _ in range(num_images)
        ])
        
        self.state_dim = state_dim
        self.state_chunk = state_chunk

        self.state_decoder = nn.Sequential(*[
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.state_dim * self.state_chunk),
            nn.ReLU(inplace=True),
        ])

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, dim)
        """
        output = {}
        for i in range(len(self.img_decoders)):
            output[f'cam_{i}'] = self.img_decoders[i](self.emb_to_img_dec(embedding).view(embedding.shape[0], 
                                                                                          self.input_channels,
                                                                                          self.img_dec_img_size,
                                                                                          self.img_dec_img_size))
        output['state'] = self.state_decoder(embedding).view(embedding.shape[0], self.state_chunk, self.state_dim)
        return output

class ResNet34Decoder(nn.Module):
    """
    Structural mirror of ResNet34 encoder for reconstruction.

    Input:  (B, 512, 7, 7)  [typical ResNet34 bottleneck feature map for 224x224 input]
    Output: (B, out_channels, 224, 224)

    Key mirror property:
      - Within each stage, we do (blocks-1) same-res blocks first,
        then ONE upsample block at the end (mirror of encoder downsample-first).
    """
    def __init__(self, out_channels=3, norm_layer=nn.BatchNorm2d, final_activation="sigmoid"):
        super().__init__()
        self.in_channels = 512

        # Mirror of encoder layer4(3), layer3(6), layer2(4), layer1(3)
        # but arranged so that the upsample happens at the END of each stage.
        self.layer4 = self._make_stage(out_channels=256, blocks=3, upsample=True,  norm_layer=norm_layer)  # 7 -> 14
        self.layer3 = self._make_stage(out_channels=128, blocks=6, upsample=True,  norm_layer=norm_layer)  # 14 -> 28
        self.layer2 = self._make_stage(out_channels=64,  blocks=4, upsample=True,  norm_layer=norm_layer)  # 28 -> 56
        self.layer1 = self._make_stage(out_channels=64,  blocks=3, upsample=False, norm_layer=norm_layer)  # 56 -> 56

        # Mirror-ish stem reversal: maxpool(s2,k3,p1) then conv7x7(s2,p3)
        self.depool = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )
        self.deconv1 = nn.ConvTranspose2d(
            64, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1, bias=True
        )

        if final_activation in ("sigmoid", "Sigmoid"):
            self.out_act = nn.Sigmoid()
        elif final_activation in ("tanh", "Tanh"):
            self.out_act = nn.Tanh()
        elif final_activation in (None, "none", "identity", "Identity"):
            self.out_act = nn.Identity()
        elif final_activation in ("softmax"):
            self.out_act = nn.Softmax2d()
        else:
            raise ValueError(f"Unsupported final_activation: {final_activation}")

    def _make_stage(self, out_channels, blocks, upsample, norm_layer):
        layers = []

        # (blocks-1) blocks at current resolution / channels
        for _ in range(blocks - 1):
            layers.append(DecoderBasicBlock(self.in_channels, self.in_channels, upsample=False, norm_layer=norm_layer))

        # Final block: either upsample+change channels, or keep as same-res block
        layers.append(DecoderBasicBlock(self.in_channels, out_channels, upsample=upsample, norm_layer=norm_layer))
        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, 512, 4, 4)
        """
        x = self.layer4(x)   # 7 -> 14
        x = self.layer3(x)   # 14 -> 28
        x = self.layer2(x)   # 28 -> 56
        x = self.layer1(x)   # 56 -> 56
        x = self.depool(x)   # 56 -> 112
        x = self.deconv1(x)  # 112 -> 224
        x = self.out_act(x)
        return x


class DecoderBasicBlock(nn.Module):
    """
    Mirror of ResNet BasicBlock, but with optional upsampling using ConvTranspose2d.

    If upsample=True:
      - conv1 is a 3x3 ConvTranspose2d with stride=2 (upsample by 2)
      - shortcut also upsamples by 2 via 1x1 ConvTranspose2d

    If upsample=False:
      - conv1 is a standard 3x3 Conv2d stride=1
      - shortcut is identity unless channel mismatch
    """
    def __init__(self, in_channels, out_channels, upsample: bool, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        if upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )

        self.bn1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)

        # Shortcut
        if upsample:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=2, output_padding=1, bias=False
                ),
                norm_layer(out_channels)
            )
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                norm_layer(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + residual)
        return out
