# src/models.py

"""
PyTorch implementations of Convolutional Neural Networks for MRI reconstruction.

This module provides:
- MiniUNet: A lightweight U-Net style architecture.
- ResMiniUNet: A MiniUNet with a residual connection from input to output.
"""

import torch
import torch.nn as nn

class MiniUNet(nn.Module):
    """
    A lightweight U-Net style architecture for MRI reconstruction.
    """
    def __init__(self, img_size=256):
        super(MiniUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(2, 32, 3, padding=1)  # Input: real+imag channels
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 256, 3, padding=1)
        
        # Decoder
        self.dec3 = nn.Conv2d(384, 128, 3, padding=1) # 256 from upsample + 128 from skip
        self.dec2 = nn.Conv2d(192, 64, 3, padding=1)  # 128 from upsample + 64 from skip
        self.dec1 = nn.Conv2d(96, 32, 3, padding=1)   # 64 from upsample + 32 from skip
        self.out = nn.Conv2d(32, 1, 3, padding=1)     # Output: 1-channel real image
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))
        
        # Bottleneck
        b = self.relu(self.bottleneck(self.pool(e3)))
        
        # Decoder with skip connections
        d3 = self.relu(self.dec3(torch.cat([self.upsample(b), e3], dim=1)))
        d2 = self.relu(self.dec2(torch.cat([self.upsample(d3), e2], dim=1)))
        d1 = self.relu(self.dec1(torch.cat([self.upsample(d2), e1], dim=1)))
        
        # Output
        out = self.out(self.dropout(d1))
        
        return out

class ResMiniUNet(MiniUNet):
    """
    MiniUNet with an added residual connection from input to output.
    This helps the network learn the residual (the artifacts) instead of the full image.
    """
    def __init__(self, img_size=512):
        super(ResMiniUNet, self).__init__(img_size)
        # 1x1 convolution to project the 2-channel input to a 1-channel residual
        self.res_conv = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        # Project the real channel of the input image as the residual base
        # Here we approximate the zero-filled image from the input k-space
        # A simple projection is used for demonstration.
        input_image_approx = torch.sqrt(x[:, 0:1, :, :]**2 + x[:, 1:2, :, :]**2)

        unet_output = super().forward(x)

        # Add the residual connection
        return unet_output + input_image_approx
