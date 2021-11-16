import torch
import torch.nn as nn


from ..outpaint_blocks import *
from ..normalizers import Normalizer
from ..activations import Activation
from ..util import *

IMG_DIMS = (256, 128)
REGION_DIMS = (40, 20)

class Generator(nn.Module):
    '''
    Args:
        latent_channels (int):  Latent channels at deepest feature layer
        add_final_conv (bool):  Adds a final conv layer after the final NS-outpaint convtranspose layer to smooth results
        upsampling_type (str):  Layer type used to upsample features to a higher resolution
    '''
    def __init__(self, latent_channels: int = 1024, add_final_conv: bool = False, upsampling_type: str = "conv_transpose"):
        super(Generator, self).__init__()

        # Normalizers and Activations shorthand
        enc_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.LRELU}
        dec_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.RELU}

        # Channel counts (ci = channel count after encoding stage i, or before decoding stage -i)
        # Bottleneck channels are generally 1/4 of stage channel count
        # (except rct, where bottleneck is 1/2 of stage channel count)
        min_channels = 32
        c0 = max(latent_channels // (2**4), min_channels)
        c1 = min(c0 * 2, latent_channels)
        c2 = min(c1 * 2, latent_channels)
        c3 = min(c2 * 2, latent_channels)
        c4 = min(c3 * 2, latent_channels)

        # Latent width
        latent_width = 4

        # Define upsampling layer type
        def upsampling_layer(in_channels: int, out_channels: int):
            if upsampling_type == "conv_transpose":
                return nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=4, stride=2, padding=1
                )

            elif upsampling_type == "nn_upsample":
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=1, padding=1
                    )
                )

            else:
                raise ValueError(f"Unrecognized upsampling type parameter: {upsampling_type}")


        # --- Encoder ---

        # Stage 0: 128 -> 64
        self.encoder_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=c0, kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c0),
            Activation.LRELU.create_layer()
        )

        # Stage 1: 64 -> 32
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=c0, out_channels=c1, kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c1),
            Activation.LRELU.create_layer()
        )

        # Stage 2: 32 -> 16
        self.encoder_2 = nn.Sequential(
            BottleneckResblock(
                c1, c2 // 4, c2, kernel_size=3, stride=2, **enc_norm_act
            ),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **enc_norm_act
            )
        )

        # Stage 3: 16 -> 8
        self.encoder_3 = nn.Sequential(
            BottleneckResblock(
                c2, c3 // 4, c3, kernel_size=3, stride=2, **enc_norm_act
            ),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **enc_norm_act
            )
        )

        # Stage 4: 8 -> 4
        self.encoder_4 = nn.Sequential(
            BottleneckResblock(
                c3, c4 // 4, c4, kernel_size=3, stride=2, **enc_norm_act
            ),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **enc_norm_act
            ),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **enc_norm_act
            )
        )

        # --- RCT Transfer ---
        self.rct = RecurrentContentTransfer(c4, c4 // 2, latent_width, **enc_norm_act)

        # --- Decoder ---

        # Stage -4: 4 -> 8
        self.decoder_4 = nn.Sequential(
            GlobalResidualBlock(c4, (7, 3), dilation=1, **dec_norm_act),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **dec_norm_act
            ),
            BottleneckResblock(
                c4, c4 // 4, c4, kernel_size=3, stride=1, **dec_norm_act
            ),
            upsampling_layer(c4, c3),
            Normalizer.INSTANCE_NORM.create_layer(c3)
        )
        self.shc_4 = SkipHorizontalConnection(c3, kernel_size=3, **dec_norm_act)

        # Stage -3: 8 -> 16
        self.decoder_3 = nn.Sequential(
            GlobalResidualBlock(c3, (7, 3), dilation=2, **dec_norm_act),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **dec_norm_act
            ),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **dec_norm_act
            ),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **dec_norm_act
            ),
            upsampling_layer(c3, c2),
            Normalizer.INSTANCE_NORM.create_layer(c2)
        )
        self.shc_3 = SkipHorizontalConnection(c2, kernel_size=3, **dec_norm_act)

        # Stage -2: 16 -> 32
        self.decoder_2 = nn.Sequential(
            GlobalResidualBlock(c2, (7, 3), dilation=4, **dec_norm_act),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **dec_norm_act
            ),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **dec_norm_act
            ),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **dec_norm_act
            ),
            upsampling_layer(c2, c1),
            Normalizer.INSTANCE_NORM.create_layer(c1)
        )
        self.shc_2 = SkipHorizontalConnection(c1, kernel_size=3, **dec_norm_act)

        # Stage -1: 32 -> 64
        self.decoder_1 = nn.Sequential(
            upsampling_layer(c1, c0),
            Normalizer.INSTANCE_NORM.create_layer(c0)
        )
        self.shc_1 = SkipHorizontalConnection(c0, kernel_size=3, **dec_norm_act)

        # Stage -0: 64 -> 128
        if add_final_conv:
            self.decoder_0 = nn.Sequential(
                upsampling_layer(c0, c0),

                # Extra conv layer after NS-outpaint implementation to smooth result
                Normalizer.INSTANCE_NORM.create_layer(c0),
                Activation.RELU.create_layer(),
                nn.Conv2d(
                    in_channels=c0, out_channels=1,
                    kernel_size=3, stride=1, padding=1
                )
            )
        else:
            self.decoder_0 = upsampling_layer(c0, 1)


    def forward(self, x):
        enc_0 = self.encoder_0(x)
        enc_1 = self.encoder_1(enc_0)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)

        dec_4 = self.rct(enc_4)

        dec_3 = self.shc_4(self.decoder_4(dec_4), enc_3)
        dec_2 = self.shc_3(self.decoder_3(dec_3), enc_2)
        dec_1 = self.shc_2(self.decoder_2(dec_2), enc_1)
        dec_0 = self.shc_1(self.decoder_1(dec_1), enc_0)
        out = self.decoder_0(dec_0)
        
        return out



'''
Use SnPatchGan style discriminator from deepfill v2
'''
class Discriminator(nn.Module):
    '''
    Args:
        latent_channels (int): Latent channels at deepest layer
    '''
    def __init__(self, local: bool, latent_channels: int = 256):
        super(Discriminator, self).__init__()

        # Set up activation
        activation = Activation.LRELU

        # Channel Counts
        min_channels = 32
        c1 = max(latent_channels // (2**2), min_channels)
        c2 = min(c1 * 2, latent_channels)
        c3 = min(c2 * 2, latent_channels)
        c4 = c3

        # Calculate final flattened features
        start_area = IMG_DIMS[0] * IMG_DIMS[1]
        if local:
            start_area = start_area // 2
        end_area = start_area // (4**4)
        end_volume = end_area * c4

        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                in_channels=1, out_channels=c1,
                kernel_size=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=c1, out_channels=c2,
                kernel_size=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=c2, out_channels=c3,
                kernel_size=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=c3, out_channels=c4,
                kernel_size=5, stride=2, padding=2
            )),
            activation.create_layer(),
            nn.Flatten(),
            nn.Linear(end_volume, 1)
        )

    def forward(self, x):
        return self.layers(x)