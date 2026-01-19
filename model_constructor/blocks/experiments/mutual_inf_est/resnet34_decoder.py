import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    """
    A 'Reverse' BasicBlock.
    - Instead of downsampling, we optionally upsample using ConvTranspose2d.
    - It maintains the residual structure (x + F(x)).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DecoderBlock, self).__init__()
        
        # If stride > 1, we use ConvTranspose2d to upsample
        if stride > 1:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut handling: if input/output dimensions or channels strictly change
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            if stride > 1:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet34Decoder(nn.Module):
    def __init__(self, n_classes=3):
        super(ResNet34Decoder, self).__init__()
        
        # Mirroring ResNet34 counts: [3, 4, 6, 3] -> Reversed: [3, 6, 4, 3]
        # Standard ResNet34 channels: [64, 128, 256, 512]
        
        # 1. Start with 512 channels (from the encoder bottleneck)
        self.in_channels = 512
        
        # Layer 4 Reverse: 512 -> 256
        self.layer4 = self._make_layer(256, blocks=3, stride=2)
        
        # Layer 3 Reverse: 256 -> 128
        self.layer3 = self._make_layer(128, blocks=6, stride=2)
        
        # Layer 2 Reverse: 128 -> 64
        self.layer2 = self._make_layer(64, blocks=4, stride=2)
        
        # Layer 1 Reverse: 64 -> 64 (No stride here usually, or stride to match stem)
        # Note: In Encoder, Layer 1 doesn't downsample. 
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        
        # Final Upsampling (Reversing the initial MaxPool and 7x7 Conv)
        # Encoder Stem: Conv 7x7 (s2) -> MaxPool (s2) = 4x reduction
        # We need 4x upsampling to get back to original size.
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # Reverses MaxPool
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, n_classes, kernel_size=4, stride=2, padding=1), # Reverses Initial Conv
            # Note: Final layer usually doesn't have BN/ReLU if it's the image output
            nn.Sigmoid() # Or Tanh, depending on your normalization
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        
        # The first block in the sequence handles the upsampling (stride) and channel change
        layers.append(DecoderBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        
        # The subsequent blocks keep the same channels and dimension
        for _ in range(1, blocks):
            layers.append(DecoderBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.final_upsample(x)
        return x

# --- Usage Example ---
if __name__ == "__main__":
    # Assume we have an input image of 224x224
    # ResNet34 Encoder output shape is typically (Batch, 512, 7, 7)
    dummy_encoder_output = torch.randn(1, 512, 7, 7)
    
    decoder = ResNet34Decoder(n_classes=3) # Output RGB
    output = decoder(dummy_encoder_output)
    
    print(f"Input Shape: {dummy_encoder_output.shape}")
    print(f"Output Shape: {output.shape}") 
    # Expected: torch.Size([1, 3, 224, 224])