from ast import increment_lineno
import torch
import torch.nn as nn


from ..outpaint_blocks import *
from normalizers import Normalizer
from activations import Activation
from ..util import *

IMG_DIMS = (64, 64)
REGION_DIMS = (20, 20)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        enc_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.LRELU}
        dec_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.RELU}

        # --- Encoder ---

        # Stage 1: 64 -> 32
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(32),
            Activation.LRELU.create_layer()
        )

        # Stage 2: 32 -> 16
        self.encoder_2 = nn.Sequential(
            BottleneckResblock(
                32, 32, 64, kernel_size=3, stride=2, **enc_norm_act
            )
        )

        # Stage 3: 16 -> 8
        self.encoder_3 = nn.Sequential(
            BottleneckResblock(
                64, 32, 128, kernel_size=3, stride=2, **enc_norm_act
            ),
            BottleneckResblock(
                128, 32, 128, kernel_size=3, stride=1, **enc_norm_act
            )
        )

        # Stage 4: 8 -> 4
        self.encoder_4 = nn.Sequential(
            BottleneckResblock(
                128, 64, 256, kernel_size=3, stride=2, **enc_norm_act
            ),
            BottleneckResblock(
                256, 64, 256, kernel_size=3, stride=1, **enc_norm_act
            )
        )

        # --- RCT Transfer ---
        self.rct = RecurrentContentTransfer(256, 128, 4, **enc_norm_act)

        # --- Decoder ---

        # Stage -4: 4 -> 8
        self.decoder_4 = nn.Sequential(
            GlobalResidualBlock(256, (7, 3), dilation=1, **dec_norm_act),
            BottleneckResblock(
                256, 64, 256, kernel_size=3, stride=1, **dec_norm_act
            ),
            nn.Conv2dTranspose(
                in_channels=256, out_channels=128,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(128)
        )
        self.shc_4 = SkipHorizontalConnection(128, kernel_size=3, **dec_norm_act)

        # Stage -3: 8 -> 16
        self.decoder_3 = nn.Sequential(
            GlobalResidualBlock(128, (7, 3), dilation=2, **dec_norm_act),
            BottleneckResblock(
                128, 32, 128, kernel_size=3, stride=1, **dec_norm_act
            ),
            nn.Conv2dTranspose(
                in_channels=128, out_channels=64,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(64)
        )
        self.shc_3 = SkipHorizontalConnection(64, kernel_size=3, **dec_norm_act)

        # Stage -2: 16 -> 32
        self.decoder_2 = nn.Sequential(
            GlobalResidualBlock(64, (7, 3), dilation=4, **dec_norm_act),
            BottleneckResblock(
                64, 32, 64, kernel_size=3, stride=1, **dec_norm_act
            ),
            nn.Conv2dTranspose(
                in_channels=64, out_channels=32,
                kernel_size=4, stride=2, padding=1
            ),
            Normalizer.INSTANCE_NORM.create_layer(32)
        )
        self.shc_2 = SkipHorizontalConnection(32, kernel_size=3, **dec_norm_act)

        # Stage -1: 32 -> 64
        self.decoder_1 = nn.Sequential(
            nn.Conv2dTranspose(
                in_channels=32, out_channels=1,
                kernel_size=4, stride=2, padding=1
            )
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
    def __init__(self):
        super(SnPatchGanDiscriminator, self).__init__()

        cnum = 32
        activation = Activation.LRELU

        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                in_channels=1, out_channels=cnum,
                kernel=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum, out_channels=cnum * 2,
                kernel=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 2, out_channels=cnum * 2,
                kernel=5, stride=2, padding=2
            )),
            activation.create_layer(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 2, out_channels=cnum * 2,
                kernel=5, stride=2, padding=2
            )),
            activation.create_layer(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)