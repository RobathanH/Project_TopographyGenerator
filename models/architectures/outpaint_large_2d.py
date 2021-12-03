from models.architectures import outpaint_large_noisy_adain
import torch
import torch.nn as nn

from typing import Tuple

from . import outpaint_large
from ..normalizers import *
from ..activations import *
from ..outpaint_blocks import *
from ..conv_blocks import *

IMG_DIMS = (256, 256)
REGION_DIMS = (40, 40)

'''
Models to generate the top-right quadrant of an image when given the other three quadrants.
Utilizes parts of the 1D outpainter models, specifically outpaint_large
'''


'''
Uses the modules from the one-dimensional generator to generate 2d images.
Default mode has no added trainable modules
'''
class Generator(outpaint_large.Generator):
    '''
    Args:
        latent_channels (int):          Latent channels at deepest feature layer
        add_final_conv (bool):          Adds a final conv layer after the final NS-outpaint convtranspose layer to smooth results
        upsampling_type (str):          Layer type used to upsample features to a higher resolution
        grb_kernel_shape (int, int):    Kernel shape for the GRB layers. Defaults to (7 width, 3 height)
    '''
    def __init__(self, latent_channels: int = 1024, add_final_conv: bool = False, upsampling_type: str = "conv_transpose", grb_kernel_shape: Tuple[int, int] = (7, 3)):
        super(Generator, self).__init__(latent_channels, add_final_conv, upsampling_type)

    '''
    Forward generation pass
    Args:
        full_image (torch.Tensor): Input images of shape (batch_size, 1, *IMG_DIMS)
    '''
    def forward(self, full_image: torch.Tensor) -> torch.Tensor:
        # Split input image into quadrants
        ll_input, lh_input, hl_input, _ = split_quadrants(full_image)

        # Encode bottom left quadrant: x low, y low
        ll_enc_0 = self.encoder_0(ll_input)
        ll_enc_1 = self.encoder_1(ll_enc_0)
        ll_enc_2 = self.encoder_2(ll_enc_1)
        ll_enc_3 = self.encoder_3(ll_enc_2)
        ll_latent = self.encoder_4(ll_enc_3)

        # Encode top left quadrant: x low, y high
        lh_enc_0 = self.encoder_0(lh_input)
        lh_enc_1 = self.encoder_1(lh_enc_0)
        lh_enc_2 = self.encoder_2(lh_enc_1)
        lh_enc_3 = self.encoder_3(lh_enc_2)
        lh_latent = self.encoder_4(lh_enc_3)

        # Encode bottom right quadrant: x high, y low
        hl_enc_0 = self.encoder_0(hl_input)
        hl_enc_1 = self.encoder_1(hl_enc_0)
        hl_enc_2 = self.encoder_2(hl_enc_1)
        hl_enc_3 = self.encoder_3(hl_enc_2)
        hl_latent = self.encoder_4(hl_enc_3)



        # Generate latent image in unseen quadrant by averaging latent images generated
        # by extending from latent images in adjacent seen quadrants
        hl_latent_rot_270 = rotate_90_CW(hl_latent)
        hh_latent_from_hl_rot_270 = self.rct.generate(hl_latent_rot_270)
        hh_latent_from_hl = rotate_90_CCW(hh_latent_from_hl_rot_270)

        hh_latent_from_lh = self.rct.generate(lh_latent)

        hh_latent = (hh_latent_from_hl + hh_latent_from_lh) / 2



        # Generate image from the latent dimensions of each quadrant, including
        # skip-horizontal-connections to process intermediate features of each known-quadrant's
        # encoding
        full_latent = combine_quadrants(ll_latent, lh_latent, hl_latent, hh_latent)
        
        # Full decoder layer
        full_dec_3 = self.decoder_4(full_latent)
        # Quadrant-wise SHC layer
        ll_dec_3, lh_dec_3, hl_dec_3, hh_dec_3 = split_quadrants(full_dec_3)
        ll_dec_3 = self.shc_4.combine(ll_dec_3, ll_enc_3)
        lh_dec_3 = self.shc_4.combine(lh_dec_3, lh_enc_3)
        hl_dec_3 = self.shc_4.combine(hl_dec_3, hl_enc_3)
        full_dec_3 = combine_quadrants(ll_dec_3, lh_dec_3, hl_dec_3, hh_dec_3)

        # Full decoder layer
        full_dec_2 = self.decoder_3(full_dec_3)
        # Quadrant-wise SHC layer
        ll_dec_2, lh_dec_2, hl_dec_2, hh_dec_2 = split_quadrants(full_dec_2)
        ll_dec_2 = self.shc_3.combine(ll_dec_2, ll_enc_2)
        lh_dec_2 = self.shc_3.combine(lh_dec_2, lh_enc_2)
        hl_dec_2 = self.shc_3.combine(hl_dec_2, hl_enc_2)
        full_dec_2 = combine_quadrants(ll_dec_2, lh_dec_2, hl_dec_2, hh_dec_2)

        # Full decoder layer
        full_dec_1 = self.decoder_2(full_dec_2)
        # Quadrant-wise SHC layer
        ll_dec_1, lh_dec_1, hl_dec_1, hh_dec_1 = split_quadrants(full_dec_1)
        ll_dec_1 = self.shc_2.combine(ll_dec_1, ll_enc_1)
        lh_dec_1 = self.shc_2.combine(lh_dec_1, lh_enc_1)
        hl_dec_1 = self.shc_2.combine(hl_dec_1, hl_enc_1)
        full_dec_1 = combine_quadrants(ll_dec_1, lh_dec_1, hl_dec_1, hh_dec_1)

        # Full decoder layer
        full_dec_0 = self.decoder_1(full_dec_1)
        # Quadrant-wise SHC layer
        ll_dec_0, lh_dec_0, hl_dec_0, hh_dec_0 = split_quadrants(full_dec_0)
        ll_dec_0 = self.shc_1.combine(ll_dec_0, ll_enc_0)
        lh_dec_0 = self.shc_1.combine(lh_dec_0, lh_enc_0)
        hl_dec_0 = self.shc_1.combine(hl_dec_0, hl_enc_0)
        full_dec_0 = combine_quadrants(ll_dec_0, lh_dec_0, hl_dec_0, hh_dec_0)

        # Final full decoder layer
        output = self.decoder_0(full_dec_0)

        return output

'''
Adds additional Noisy Ada In layers. NoisyAdaIn layers must upsample to twice the vertical size,
requiring new init function
'''
class GeneratorNoisyAdaIn(Generator):
    '''
    Args:
        latent_channels (int):          Latent channels at deepest feature layer
        add_final_conv (bool):          Adds a final conv layer after the final NS-outpaint convtranspose layer to smooth results
        upsampling_type (str):          Layer type used to upsample features to a higher resolution
        min_channels (int):             Minimum channels at earliest and latest non-image feature layers
        grb_kernel_shape (int, int):    Kernel shape for the GRB layers. Defaults to (7 width, 3 height)
    '''
    def __init__(self, latent_channels: int = 1024, add_final_conv: bool = False, upsampling_type: str = "conv_transpose", min_channels: int = 32, grb_kernel_shape: Tuple[int, int] = (7, 3)):
        super(Generator, self).__init__()

        # Normalizers and Activations shorthand
        enc_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.LRELU}
        dec_norm_act = {'normalizer': Normalizer.INSTANCE_NORM, 'activation': Activation.RELU}

        # Channel counts (ci = channel count after encoding stage i, or before decoding stage -i)
        # Bottleneck channels are generally 1/4 of stage channel count
        # (except rct, where bottleneck is 1/2 of stage channel count)
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
            GlobalResidualBlock(c4, grb_kernel_shape, dilation=1, **dec_norm_act),
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
        self.ada_4 = NoisyAdaIn((4 * latent_width, 4 * latent_width), c3, latent_channels, 1, id_init=True, activation=Activation.NONE)

        # Stage -3: 8 -> 16
        self.decoder_3 = nn.Sequential(
            GlobalResidualBlock(c3, grb_kernel_shape, dilation=2, **dec_norm_act),
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
        self.ada_3 = NoisyAdaIn((8 * latent_width, 8 * latent_width), c2, latent_channels, 1, id_init=True, activation=Activation.NONE)

        # Stage -2: 16 -> 32
        self.decoder_2 = nn.Sequential(
            GlobalResidualBlock(c2, grb_kernel_shape, dilation=4, **dec_norm_act),
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
        self.ada_2 = NoisyAdaIn((16 * latent_width, 16 * latent_width), c1, latent_channels, 1, id_init=True, activation=Activation.NONE)

        # Stage -1: 32 -> 64
        self.decoder_1 = nn.Sequential(
            upsampling_layer(c1, c0),
            Normalizer.INSTANCE_NORM.create_layer(c0)
        )
        self.shc_1 = SkipHorizontalConnection(c0, kernel_size=3, **dec_norm_act)
        self.ada_1 = NoisyAdaIn((32 * latent_width, 32 * latent_width), c0, latent_channels, 1, id_init=True, activation=Activation.NONE)

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

    '''
    Forward generation pass
    Args:
        full_image (torch.Tensor): Input images of shape (batch_size, 1, *IMG_DIMS)
    '''
    def forward(self, full_image: torch.Tensor) -> torch.Tensor:
        # Split input image into quadrants
        ll_input, lh_input, hl_input, _ = split_quadrants(full_image)

        # Encode bottom left quadrant: x low, y low
        ll_enc_0 = self.encoder_0(ll_input)
        ll_enc_1 = self.encoder_1(ll_enc_0)
        ll_enc_2 = self.encoder_2(ll_enc_1)
        ll_enc_3 = self.encoder_3(ll_enc_2)
        ll_latent = self.encoder_4(ll_enc_3)

        # Encode top left quadrant: x low, y high
        lh_enc_0 = self.encoder_0(lh_input)
        lh_enc_1 = self.encoder_1(lh_enc_0)
        lh_enc_2 = self.encoder_2(lh_enc_1)
        lh_enc_3 = self.encoder_3(lh_enc_2)
        lh_latent = self.encoder_4(lh_enc_3)

        # Encode bottom right quadrant: x high, y low
        hl_enc_0 = self.encoder_0(hl_input)
        hl_enc_1 = self.encoder_1(hl_enc_0)
        hl_enc_2 = self.encoder_2(hl_enc_1)
        hl_enc_3 = self.encoder_3(hl_enc_2)
        hl_latent = self.encoder_4(hl_enc_3)



        # Generate latent image in unseen quadrant by averaging latent images generated
        # by extending from latent images in adjacent seen quadrants
        hl_latent_rot_270 = rotate_90_CW(hl_latent)
        hh_latent_from_hl_rot_270 = self.rct.generate(hl_latent_rot_270)
        hh_latent_from_hl = rotate_90_CCW(hh_latent_from_hl_rot_270)

        hh_latent_from_lh = self.rct.generate(lh_latent)

        hh_latent = (hh_latent_from_hl + hh_latent_from_lh) / 2



        # Generate image from the latent dimensions of each quadrant, including
        # skip-horizontal-connections to process intermediate features of each known-quadrant's
        # encoding
        full_latent = combine_quadrants(ll_latent, lh_latent, hl_latent, hh_latent)
        
        # Full decoder layer
        full_dec_3 = self.ada_4((self.decoder_4(full_latent), full_latent))[0]
        # Quadrant-wise SHC layer
        ll_dec_3, lh_dec_3, hl_dec_3, hh_dec_3 = split_quadrants(full_dec_3)
        ll_dec_3 = self.shc_4.combine(ll_dec_3, ll_enc_3)
        lh_dec_3 = self.shc_4.combine(lh_dec_3, lh_enc_3)
        hl_dec_3 = self.shc_4.combine(hl_dec_3, hl_enc_3)
        full_dec_3 = combine_quadrants(ll_dec_3, lh_dec_3, hl_dec_3, hh_dec_3)

        # Full decoder layer
        full_dec_2 = self.ada_3((self.decoder_3(full_dec_3), full_latent))[0]
        # Quadrant-wise SHC layer
        ll_dec_2, lh_dec_2, hl_dec_2, hh_dec_2 = split_quadrants(full_dec_2)
        ll_dec_2 = self.shc_3.combine(ll_dec_2, ll_enc_2)
        lh_dec_2 = self.shc_3.combine(lh_dec_2, lh_enc_2)
        hl_dec_2 = self.shc_3.combine(hl_dec_2, hl_enc_2)
        full_dec_2 = combine_quadrants(ll_dec_2, lh_dec_2, hl_dec_2, hh_dec_2)

        # Full decoder layer
        full_dec_1 = self.ada_2((self.decoder_2(full_dec_2), full_latent))[0]
        # Quadrant-wise SHC layer
        ll_dec_1, lh_dec_1, hl_dec_1, hh_dec_1 = split_quadrants(full_dec_1)
        ll_dec_1 = self.shc_2.combine(ll_dec_1, ll_enc_1)
        lh_dec_1 = self.shc_2.combine(lh_dec_1, lh_enc_1)
        hl_dec_1 = self.shc_2.combine(hl_dec_1, hl_enc_1)
        full_dec_1 = combine_quadrants(ll_dec_1, lh_dec_1, hl_dec_1, hh_dec_1)

        # Full decoder layer
        full_dec_0 = self.ada_1((self.decoder_1(full_dec_1), full_latent))[0]
        # Quadrant-wise SHC layer
        ll_dec_0, lh_dec_0, hl_dec_0, hh_dec_0 = split_quadrants(full_dec_0)
        ll_dec_0 = self.shc_1.combine(ll_dec_0, ll_enc_0)
        lh_dec_0 = self.shc_1.combine(lh_dec_0, lh_enc_0)
        hl_dec_0 = self.shc_1.combine(hl_dec_0, hl_enc_0)
        full_dec_0 = combine_quadrants(ll_dec_0, lh_dec_0, hl_dec_0, hh_dec_0)

        # Final full decoder layer
        output = self.decoder_0(full_dec_0)

        return output

'''
Use outpaint_large discriminator architectures
Hardcode img dimensions to match this setting
'''
class LocalDiscriminator(outpaint_large.Discriminator):
    def __init__(self, **kargs):
        kargs['input_dims'] = (IMG_DIMS[0] // 2, IMG_DIMS[1] // 2)
        super(LocalDiscriminator, self).__init__(**kargs)

class GlobalDiscriminator(outpaint_large.Discriminator):
    def __init__(self, **kargs):
        kargs['input_dims'] = IMG_DIMS
        super(GlobalDiscriminator, self).__init__(**kargs)



# --- Helper Functions ---
def rotate_90_CW(input: torch.Tensor) -> torch.Tensor:
    return input.flip(2).transpose(2, 3) # new x = old y, new y = old x reversed

def rotate_90_CCW(input: torch.Tensor) -> torch.Tensor:
    return input.transpose(2, 3).flip(2)

def combine_quadrants(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    low_x = torch.cat([ll, lh], dim=3)
    high_x = torch.cat([hl, hh], dim=3)
    full = torch.cat([low_x, high_x], dim=2)
    return full

def split_quadrants(full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    low_x, high_x = torch.tensor_split(full, 2, dim=2)
    ll, lh = torch.tensor_split(low_x, 2, dim=3)
    hl, hh = torch.tensor_split(high_x, 2, dim=3)
    return ll, lh, hl, hh