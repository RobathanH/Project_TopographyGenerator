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
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=3, stride=1, padding=1
            ),
            nn.Tanh(),
            ResBlock(32, 3),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            ),
            ResBlock(64, 3),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5)
        )

        self.to_mean = nn.Conv2d(
            in_channels=128, out_channels=LATENT_CHANNELS,
            kernel_size=1, stride=1, padding=0
        )
        self.to_log_var = nn.Conv2d(
            in_channels=128, out_channels=LATENT_CHANNELS,
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
                in_channels=LATENT_CHANNELS, out_channels=64,
                kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
            ResBlock(64, 5),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
            ResBlock(128, 5),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(128, 5),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(128, 3),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(128, 3),
            nn.Conv2d(
                in_channels=128, out_channels=1,
                kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=3, stride=1, padding=1
            ),
            nn.Tanh(),
            ResBlock(32, 3),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            ),
            ResBlock(64, 3),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=5, stride=1, padding=2
            ),
            ResBlock(128, 5),
            nn.Conv2d(
                in_channels=128, out_channels=16,
                kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)