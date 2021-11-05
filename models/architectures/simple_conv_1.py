import torch
import torch.nn as nn
from ..conv_blocks import *

IMG_DIMS = (64, 64)
REGION_DIMS = (10, 10)
LATENT_DIMS = (4, 4)
LATENT_CHANNELS = 16

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            ConvPoolBlock(1, 32, 3, 2),
            ConvPoolBlock(32, 64, 3, 2),
            ConvPoolBlock(64, 128, 5, 2),
            ConvPoolBlock(128, 256, 5, 2)
        )

        self.to_mean = nn.Conv2d(
            in_channels=256, out_channels=LATENT_CHANNELS,
            kernel_size=1, stride=1, padding=0
        )
        self.to_log_var = nn.Conv2d(
            in_channels=256, out_channels=LATENT_CHANNELS,
            kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, x):
        x = self.layers(x)
        z_mean = self.to_mean(x)
        z_log_var = self.to_log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=LATENT_CHANNELS, out_channels=256,
                kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
            ReverseConvPoolBlock(256, 128, 5, 2),
            ReverseConvPoolBlock(128, 64, 5, 2),
            ReverseConvPoolBlock(64, 32, 3, 2),
            ReverseConvPoolBlock(32, 1, 3, 2, final_activation=False)
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            ConvPoolBlock(1, 32, 3, 2),
            ConvPoolBlock(32, 64, 3, 2),
            ConvPoolBlock(64, 128, 5, 2),
            ConvPoolBlock(128, 256, 5, 2),
            nn.Conv2d(
                in_channels=256, out_channels=16,
                kernel_size=1, stride=1, padding=0
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)