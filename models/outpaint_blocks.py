from typing import List, Tuple

import torch
import torch.nn as nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


from models.activations import Activation
from models.normalizers import Normalizer


'''
Larger version of a resblock, with one conv layer surrounded by two
1x1 conv layers to temporarily reduce features to a bottleneck,
with a skip connection bypassing all of these layers.
Adapted from github.com/z-x-yang/NS-Outpainting
'''
class BottleneckResblock(nn.Module):
    '''
    Args:
        in_channels (int):          Number of channels in module input
        bottleneck_channels (int):  Number of channels in bottleneck
        out_channels (int):         Number of channels in module output
        kernel_size (int):          Width of kernel in inner conv layer
        stride (int):               Effective stride of the overall module
                                    (if stride != 1, shortcut connection goes through
                                    a 1x1 strided conv block)
        activations (Activation):   Activation type between each layer
        normalizer (Normalizer):    Normalization type between each layer

    '''
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, kernel_size: int, stride: int = 1, normalizer: Normalizer = Normalizer.NONE, activation: Activation = Activation.NONE):
        super(BottleneckResblock, self).__init__()

        self.bottleneck_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=bottleneck_channels,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(bottleneck_channels),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=bottleneck_channels, out_channels=bottleneck_channels,
                kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
            ),
            normalizer.create_layer(bottleneck_channels),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=bottleneck_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(out_channels)
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut_layers = nn.Identity()
        else:
            self.shortcut_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, padding=0
                ),
                normalizer.create_layer(out_channels)
            )

        self.final_activation = activation.create_layer()

    def forward(self, x):
        bottleneck_branch = self.bottleneck_layers(x)
        shortcut_branch = self.shortcut_layers(x)
        return self.final_activation(bottleneck_branch + shortcut_branch)


'''
Global Residual Block: Similar to a ResBlock but with a large, biased receptive field
Adapted from github.com/z-x-yang/NS-Outpainting
'''
class GlobalResidualBlock(nn.Module):
    '''
    Args:
        channels (int):             Number of channels in both input and output
        kernel_shape (int, int):    Kernel size in each spatial dimension
        dilation (int):             Dilation rate for all conv layers
        activations (Activation):   Activation type between each layer
        normalizer (Normalizer):    Normalization type between each layer
    '''
    def __init__(self, channels: int, kernel_shape: Tuple[int, int], dilation: int = 1, normalizer: Normalizer = Normalizer.NONE, activation: Activation = Activation.NONE):
        super(GlobalResidualBlock, self).__init__()

        padding_shape = dilation * (kernel_shape[0] - 1) // 2, dilation * (kernel_shape[1] - 1) // 2

        self.branch_xy = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=(kernel_shape[0], 1), stride=1, padding=(padding_shape[0], 0), dilation=dilation
            ),
            normalizer.create_layer(channels),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=(1, kernel_shape[1]), stride=1, padding=(0, padding_shape[1]), dilation=dilation
            ),
            normalizer.create_layer(channels)
        )

        self.branch_yx = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=(1, kernel_shape[1]), stride=1, padding=(0, padding_shape[1]), dilation=dilation
            ),
            normalizer.create_layer(channels),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=(kernel_shape[0], 1), stride=1, padding=(padding_shape[0], 0), dilation=dilation
            ),
            normalizer.create_layer(channels)
        )

        self.final_activation = activation.create_layer()

    def forward(self, input):
        return self.final_activation(input + self.branch_xy(input) + self.branch_yx(input))


'''
Skip Horizontal Connection Block:
Takes partially encoded features and half of the partially decoded
image, concatenates them, applies a bottle-necked conv layer,
then adds the result back into the encoded features.
Both input sources must have the same shape and channel count.
Adapted from github.com/z-x-yang/NS-Outpainting
'''
class SkipHorizontalConnection(nn.Module):
    '''
    Args:
        channels (int):             Number of channels in both input and output
        kernel_size (int):          Kernel size for bottlenecked conv layer
        activations (Activation):   Activation type between each layer
        normalizer (Normalizer):    Normalization type between each layer
    '''
    def __init__(self, channels: int, kernel_size: int = 3, normalizer: Normalizer = Normalizer.NONE, activation: Activation = Activation.NONE):
        super(SkipHorizontalConnection, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * channels, out_channels=channels // 2,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(channels // 2),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=channels // 2, out_channels=channels // 2,
                kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
            normalizer.create_layer(channels // 2),
            activation.create_layer(),
            nn.Conv2d(
                in_channels=channels // 2, out_channels=channels,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(channels)
        )

        self.final_norm = normalizer.create_layer(channels)
        self.final_activation = activation.create_layer()

    '''
    Processes a portion of the partially decoded features by combining
    them with spatially-corresponding partially encoded features.
    Performs entire skip horizontal connection without the stitching step.
    Inputs from encoder and decoder are assumed to have the same size,
    and correspond to the same region of the image.
    Args:
        from_decoder (torch.Tensor)
        from_encoder (torch.Tensor)
    '''
    def combine(self, from_decoder, from_encoder):
        combined = torch.cat([from_decoder, from_encoder], dim=1)
        combined = self.layers(combined)
        combined = combined + from_encoder
        combined = self.final_norm(combined)
        combined = self.final_activation(combined)
        return combined


    '''
    Args:
        x:              Partially decoded features
        left_shortcut:  The partially encoded features from the source image 
                        (left half of generated region)
    '''
    def forward(self, x, left_shortcut):
        left_x, right_x = torch.tensor_split(x, 2, dim=2)
        left_combined = self.combine(left_x, left_shortcut)
        stitched = torch.cat([left_combined, right_x], dim=2)
        return stitched


'''
Recurrent Content Transfer Block. Uses an LSTM to extend latent features into
an unseen area for image generation. Expands in the positive x direction.
Adapted from github.com/z-x-yang/NS-Outpainting
'''
class RecurrentContentTransfer(nn.Module):
    '''
    Args:
        channels (int):                 Number of channels for input and output
        bottleneck_channels (int):      Number of channels between 1x1 bottlenecks
        input_width (int):              Spatial width/height of the feature volume
        activations (Activation):       Activation type between each layer
        normalizer (Normalizer):        Normalization type between each layer
    '''
    def __init__(self, channels: int, bottleneck_channels: int, input_width: int, normalizer: Normalizer = Normalizer.NONE, activation: Activation = Activation.NONE):
        super(RecurrentContentTransfer, self).__init__()

        self.input_width = input_width
        self.lstm_features = bottleneck_channels * input_width

        self.bottleneck_in = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=bottleneck_channels,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(bottleneck_channels),
            activation.create_layer()
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_features, hidden_size=self.lstm_features, num_layers=2
        )
        
        self.bottleneck_out = nn.Sequential(
            nn.Conv2d(
                in_channels=bottleneck_channels, out_channels=channels,
                kernel_size=1, stride=1, padding=0
            ),
            normalizer.create_layer(channels),
            activation.create_layer()
        )

    '''
    Uses the LSTM layers to generate new latent image adjacent to
    the given latent image, but doesn't stitch together with given latent
    image
    Args:
        latent (torch.Tensor)
    Returns:
        generated_adjacent_latent (torch.Tensor)
    '''
    def generate(self, latent):
        # Reduce channel count into bottleneck
        bottlenecked = self.bottleneck_in(latent) # N, C, W, H
        N, C, W, H = bottlenecked.shape

        # Rearrange into lstm format
        reordered_in = bottlenecked.permute(0, 2, 3, 1) # N, W, H, C
        flattened_in = reordered_in.reshape(N, W, -1) # N, W, H*C
        sequenced_in = flattened_in.permute(1, 0, 2) # W, N, H*C

        # Pass through LSTM
        out, (h, c) = self.lstm(sequenced_in)
        sequenced_out = torch.zeros(self.input_width, latent.shape[0], self.lstm_features).to(DEVICE) # W, N, H*C
        sequenced_out[0] = out[-1]
        for step in range(1, self.input_width):
            out, (h, c) = self.lstm(out[-1:], (h, c))
            sequenced_out[step] = out[-1]

        # Rearrange for conv layers
        flattened_out = sequenced_out.permute(1, 0, 2) # N, W, H*C
        reordered_out = flattened_out.reshape(N, W, H, C) # N, W, H, C
        bottlenecked_out = reordered_out.permute(0, 3, 1, 2) # N, C, W, H
        
        # Increase channel count out of bottleneck
        out = self.bottleneck_out(bottlenecked_out)

        return out

    '''
    Extends the given latent image to the right with newly generated
    latent features.
    Args:
        latent (torch.Tensor)
    '''
    def forward(self, latent):
        generated_latent = self.generate(latent)

        # Stitch together input and produced features
        stitched = torch.cat([latent, generated_latent], dim=2)

        return stitched

