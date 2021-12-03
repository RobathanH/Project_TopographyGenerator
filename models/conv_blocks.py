from typing import Tuple
import torch
import torch.nn as nn

from .activations import Activation

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def ConvPoolBlock(input_channels, output_channels, kernel_size, pool_factor):
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.Tanh(),
        nn.Conv2d(
            in_channels=output_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.MaxPool2d(
            kernel_size=pool_factor, stride=pool_factor, padding=0
        ),
        nn.Tanh()
    )

def ReverseConvPoolBlock(input_channels, output_channels, kernel_size, size_factor, final_activation=True):
    padding = (kernel_size - 1) // 2

    layers = [
        nn.Upsample(
            scale_factor=size_factor, mode='bilinear', align_corners=True
        ),
        nn.Conv2d(
            in_channels=input_channels, out_channels=input_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.Tanh(),
        nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
    ]
    if final_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )
        )

    def forward(self, x):
        z = self.layers(x)
        out = nn.ReLU()(x + z)
        return out


class NoisyAdaIn(nn.Module):
    '''
    Args:
        input_shape ((int, int))
        channels (int)
        latent_channels (int)
        kernel_size (int)
        id_init (bool):             Whether to initialize all layers with constants such that initial operation is identity transform
        activation (Activation):    Activation function to use
    '''
    def __init__(self, input_shape: Tuple[int, int], channels: int, latent_channels: int, kernel_size: int = 5, id_init: bool = False, activation: Activation = Activation.TANH) -> None:
        super(NoisyAdaIn, self).__init__()

        self.input_shape = input_shape
        self.channels = channels

        self.noise_scale = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.to_mean = nn.Conv2d(
            in_channels=latent_channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, padding=2
        )
        self.to_scale = nn.Conv2d(
            in_channels=latent_channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, padding=2
        )
        self.activation = activation.create_layer()

        # Force constant initialization
        if id_init:
            nn.init.constant_(self.noise_scale, 0)
            nn.init.constant_(self.to_mean.weight, 0)
            nn.init.constant_(self.to_mean.bias, 0)
            nn.init.constant_(self.to_scale.weight, 0)
            nn.init.constant_(self.to_scale.bias, 1)

    def forward(self, x):
        # Unpack inputs
        x, z = x

        # Add noise
        noise1 = self.noise_scale * torch.randn(1, self.channels, *self.input_shape).to(DEVICE)
        x = x + noise1
        
        # Normalize
        x = nn.InstanceNorm2d(self.channels)(x)

        # Find new per-channel mean and std from latent state
        mean = nn.Upsample(self.input_shape, mode='bilinear', align_corners=True)(self.to_mean(z))
        scale = nn.Upsample(self.input_shape, mode='bilinear', align_corners=True)(self.to_scale(z))

        x = mean + scale * x

        # Nonlinearity
        x = self.activation(x)

        return x, z


class StyleGanBlock(nn.Module):
    def __init__(self, input_shape, channels, latent_channels, kernel_size):
        super(StyleGanBlock, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
        self.ada1 = NoisyAdaIn(input_shape, channels, latent_channels)
        self.conv2 = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
        self.ada2 = NoisyAdaIn(input_shape, channels, latent_channels)

    def forward(self, x):
        # Unpack inputs
        x, z = x

        x = self.conv1(x)
        x, _ = self.ada1((x, z))
        x = self.conv2(x)
        x, _ = self.ada2((x, z))

        return x, z



class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, normalization=None, activation=None):
        super(GatedConv2d, self).__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.mask = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

        if normalization is None:
            self.normalization = None
        elif normalization == "in":
            self.normalization = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            raise NotImplementedError(f"Unrecognized normalization argument: {normalization}")

        if activation is None:
            self.activation = None
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation == nn.LeakyReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x) * torch.sigmoid(self.mask(x))

        if self.normalization is not None:
            out = self.normalization(out)

        if self.activation is not None:
            out = self.activation(out)

        return out