import torch
import torch.nn as nn


from ..outpaint_blocks import *
from ..conv_blocks import *
from ..util import *
from ..normalizers import Normalizer
from ..activations import Activation


'''
Similar to architecture in outpaint_large, but with NoisyAdaIn layers included
'''


IMG_DIMS = (256, 128)
REGION_DIMS = (40, 20)

class Generator(nn.Module):
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

        self.latent_channels = latent_channels
        self.latent_width = 4

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
        self.rct = RecurrentContentTransfer(c4, c4 // 2, self.latent_width, **enc_norm_act)

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
        self.ada_4 = NoisyAdaIn((4 * self.latent_width, 2 * self.latent_width), c3, latent_channels, 1, id_init=True, activation=Activation.NONE)

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
        self.ada_3 = NoisyAdaIn((8 * self.latent_width, 4 * self.latent_width), c2, latent_channels, 1, id_init=True, activation=Activation.NONE)

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
        self.ada_2 = NoisyAdaIn((16 * self.latent_width, 8 * self.latent_width), c1, latent_channels, 1, id_init=True, activation=Activation.NONE)

        # Stage -1: 32 -> 64
        self.decoder_1 = nn.Sequential(
            upsampling_layer(c1, c0),
            Normalizer.INSTANCE_NORM.create_layer(c0)
        )
        self.shc_1 = SkipHorizontalConnection(c0, kernel_size=3, **dec_norm_act)
        self.ada_1 = NoisyAdaIn((32 * self.latent_width, 16 * self.latent_width), c0, latent_channels, 1, id_init=True, activation=Activation.NONE)

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
    Perform a full outpainting pass, removing right half of image,
    and replacing it with generated extension to given left half
    '''
    def forward(self, full_image: torch.Tensor) -> torch.Tensor:
        # Remove right half of image
        input = torch.tensor_split(full_image, 2, dim=2)[0]

        enc_0 = self.encoder_0(input)
        enc_1 = self.encoder_1(enc_0)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)

        dec_4 = self.rct(enc_4)

        dec_3 = self.ada_4((self.shc_4(self.decoder_4(dec_4), enc_3), dec_4))[0]
        dec_2 = self.ada_3((self.shc_3(self.decoder_3(dec_3), enc_2), dec_4))[0]
        dec_1 = self.ada_2((self.shc_2(self.decoder_2(dec_2), enc_1), dec_4))[0]
        dec_0 = self.ada_1((self.shc_1(self.decoder_1(dec_1), enc_0), dec_4))[0]
        out = self.decoder_0(dec_0)
        
        return out

'''
Version of noisy adain generator where ada layers are added before skip horizontal connection
'''
class Generator2(Generator):
    def forward(self, full_image: torch.Tensor) -> torch.Tensor:
          # Remove right half of image
        input = torch.tensor_split(full_image, 2, dim=2)[0]

        enc_0 = self.encoder_0(input)
        enc_1 = self.encoder_1(enc_0)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)

        dec_4 = self.rct(enc_4)

        dec_3 = self.shc_4(self.ada_4((self.decoder_4(dec_4), dec_4))[0], enc_3)
        dec_2 = self.shc_3(self.ada_3((self.decoder_3(dec_3), dec_4))[0], enc_2)
        dec_1 = self.shc_2(self.ada_2((self.decoder_2(dec_2), dec_4))[0], enc_1)
        dec_0 = self.shc_1(self.ada_1((self.decoder_1(dec_1), dec_4))[0], enc_0)
        out = self.decoder_0(dec_0)
        
        return out

    '''
    Generate a new image from a given latent representation, without the SHC layers which
    require information about the encoding process.
    Args:
        latent (torch.Tensor):  Latent representations of windows to generate in isolation.
                                Shape = (batch_size, self.latent_channels, self.latent_width, self.latent_width)
    '''
    def generate_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # Ada layers are hardcoded to apply to 2 windows side-by-side (one as input, one as generated),
        # so we have to create a fake left-image-side to use which doesn't change channel mean/variance
        # statistics. Thus, we will duplicate generated decoding and stitch two copies of it side-by-side
        # as Ada layer input.
        # To avoid forcing features on the left side of the generated image to be contiguous with features
        # on the right side of the image, we will undo this duplication step whenever we pass through a
        # convolutional module.
        pre_ada_dec_3 = self.decoder_4(latent)
        pre_ada_dec_3 = torch.cat([pre_ada_dec_3, pre_ada_dec_3], dim=2)
        post_ada_dec_3 = self.ada_4((pre_ada_dec_3, latent))[0]
        dec_3 = torch.tensor_split(post_ada_dec_3, 2, dim=2)[1]

        pre_ada_dec_2 = self.decoder_3(dec_3)
        pre_ada_dec_2 = torch.cat([pre_ada_dec_2, pre_ada_dec_2], dim=2)
        post_ada_dec_2 = self.ada_3((pre_ada_dec_2, latent))[0]
        dec_2 = torch.tensor_split(post_ada_dec_2, 2, dim=2)[1]

        pre_ada_dec_1 = self.decoder_2(dec_2)
        pre_ada_dec_1 = torch.cat([pre_ada_dec_1, pre_ada_dec_1], dim=2)
        post_ada_dec_1 = self.ada_2((pre_ada_dec_1, latent))[0]
        dec_1 = torch.tensor_split(post_ada_dec_1, 2, dim=2)[1]

        pre_ada_dec_0 = self.decoder_1(dec_1)
        pre_ada_dec_0 = torch.cat([pre_ada_dec_0, pre_ada_dec_0], dim=2)
        post_ada_dec_0 = self.ada_1((pre_ada_dec_0, latent))[0]
        dec_0 = torch.tensor_split(post_ada_dec_0, 2, dim=2)[1]

        out = self.decoder_0(dec_0)
        return out


'''
Discriminator used in NS-outpaint
'''
class Discriminator(nn.Module):
    '''
    Args:
        latent_channels (int): Latent channels at deepest layer
    '''
    def __init__(self, latent_channels: int = 256, input_dims: Tuple[int, int] = IMG_DIMS):
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
        start_area = input_dims[0] * input_dims[1]
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


'''
Hardcode img dimensions for local and global discriminator
'''
class LocalDiscriminator(Discriminator):
    def __init__(self, **kargs):
        kargs['input_dims'] = (IMG_DIMS[0] // 2, IMG_DIMS[1])
        super(LocalDiscriminator, self).__init__(**kargs)

class GlobalDiscriminator(Discriminator):
    def __init__(self, **kargs):
        kargs['input_dims'] = IMG_DIMS
        super(GlobalDiscriminator, self).__init__(**kargs)