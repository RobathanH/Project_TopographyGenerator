import torch
import torch.nn as nn


from ..outpaint_blocks import *
from ..normalizers import Normalizer
from ..activations import Activation
from ..util import *

IMG_DIMS = (128, 64)
REGION_DIMS = (40, 20)

class Generator(nn.Module):
    '''
    Args:
        latent_channels (int):  Latent channels at deepest feature layer
        add_final_conv (bool):  Adds a final conv layer after the final NS-outpaint convtranspose layer to smooth results
    '''
    def __init__(self, latent_channels: int = 256, add_final_conv: bool = False):
        super(Generator, self).__init__()

        # Normalizers and Activations shorthand
        enc_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.LRELU}
        dec_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.RELU}

        # Channel counts (ci = channel count after encoding stage i, or before decoding stage -i)
        # Bottleneck channels are generally 1/4 of stage channel count
        # (except rct, where bottleneck is 1/2 of stage channel count)
        c1 = latent_channels // (2**3)
        c2 = latent_channels // (2**2)
        c3 = latent_channels // (2**1)
        c4 = latent_channels

        # Latent width
        latent_width = 4

        # --- Encoder ---

        # Stage 1: 64 -> 32
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=c1, kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c1),
            Activation.LRELU.create_layer()
        )

        # Stage 2: 32 -> 16
        self.encoder_2 = nn.Sequential(
            BottleneckResblock(
                c1, c2 // 4, c2, kernel_size=3, stride=2, **enc_norm_act
            )
        )

        # Stage 3: 16 -> 8
        self.encoder_3 = nn.Sequential(
            BottleneckResblock(
                c2, c3 // 4, c3, kernel_size=3, stride=2, **enc_norm_act
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
            nn.ConvTranspose2d(
                in_channels=c4, out_channels=c3,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c3)
        )
        self.shc_4 = SkipHorizontalConnection(c3, kernel_size=3, **dec_norm_act)

        # Stage -3: 8 -> 16
        self.decoder_3 = nn.Sequential(
            GlobalResidualBlock(c3, (7, 3), dilation=2, **dec_norm_act),
            BottleneckResblock(
                c3, c3 // 4, c3, kernel_size=3, stride=1, **dec_norm_act
            ),
            nn.ConvTranspose2d(
                in_channels=c3, out_channels=c2,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c2)
        )
        self.shc_3 = SkipHorizontalConnection(c2, kernel_size=3, **dec_norm_act)

        # Stage -2: 16 -> 32
        self.decoder_2 = nn.Sequential(
            GlobalResidualBlock(c2, (7, 3), dilation=4, **dec_norm_act),
            BottleneckResblock(
                c2, c2 // 4, c2, kernel_size=3, stride=1, **dec_norm_act
            ),
            nn.ConvTranspose2d(
                in_channels=c2, out_channels=c1,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(c1)
        )
        self.shc_2 = SkipHorizontalConnection(c1, kernel_size=3, **dec_norm_act)

        # Stage -1: 32 -> 64
        if add_final_conv:
            self.decoder_1 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=c1, out_channels=c1,
                    kernel_size=4, stride=2, padding=1
                ),

                # Extra conv layer after NS-outpaint implementation to smooth result
                Normalizer.INSTANCE_NORM.create_layer(c1),
                Activation.RELU.create_layer(),
                nn.Conv2d(
                    in_channels=c1, out_channels=1,
                    kernel_size=3, stride=1, padding=1
                )
            )
        else:
            self.decoder_1 = nn.ConvTranspose2d(
                in_channels=c1, out_channels=1,
                kernel_size=4, stride=2, padding=1
            )


    def forward(self, x):
        enc_1 = self.encoder_1(x)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)

        dec_4 = self.rct(enc_4)

        dec_3 = self.shc_4(self.decoder_4(dec_4), enc_3)
        dec_2 = self.shc_3(self.decoder_3(dec_3), enc_2)
        dec_1 = self.shc_2(self.decoder_2(dec_2), enc_1)
        out = self.decoder_1(dec_1)
        
        return out



'''
Use SnPatchGan style discriminator from deepfill v2
'''
class SnPatchGanDiscriminator(nn.Module):
    '''
    Args:
        latent_channels (int): Latent channels at deepest layer
    '''
    def __init__(self, latent_channels: int = 64):
        super(SnPatchGanDiscriminator, self).__init__()

        # Activation setup
        activation = Activation.LRELU

        # Channel counts
        min_channels = 32
        c1 = max(latent_channels // (2**2), min_channels)
        c2 = min(c1 * 2, latent_channels)
        c3 = min(c2 * 2, latent_channels)
        c4 = c3

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
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)