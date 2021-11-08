import torch
import torch.nn as nn
from ..conv_blocks import *
from ..util import *

IMG_DIMS = (128, 128)
REGION_DIMS = (20, 20)


'''
Simplified DeepFill v2 Generator.
Similar to original 2-stage generator implementation: https://github.com/JiahuiYu/generative_inpainting,
but without the contextual attention layer in the refinement stage.
Draws from this implementation: https://github.com/zhaoyuzhi/deepfillv2
'''
class SingleBranchGenerator(nn.Module):
    def __init__(self, base_latent_channels=48):
        super(SingleBranchGenerator, self).__init__()

        # Base channel count
        cnum = base_latent_channels

        # Stage 1 - course grained
        self.stage_1 = nn.Sequential(
            GatedConv2d(2, cnum, 5),
            GatedConv2d(cnum, 2 * cnum, 3, stride=2),
            GatedConv2d(2 * cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 4 * cnum, 3, stride=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=4),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=8),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=16),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GatedConv2d(4 * cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 2 * cnum, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GatedConv2d(2 * cnum, cnum, 3),
            GatedConv2d(cnum, cnum // 2, 3),
            GatedConv2d(cnum // 2, 1, 3, activation=None)
        )

        # Stage 2 - refinement
        self.stage_2 = nn.Sequential(
            GatedConv2d(2, cnum, 5),
            GatedConv2d(cnum, 2 * cnum, 3, stride=2),
            GatedConv2d(2 * cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 4 * cnum, 3, stride=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=4),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=8),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=16),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GatedConv2d(4 * cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 2 * cnum, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GatedConv2d(2 * cnum, cnum, 3),
            GatedConv2d(cnum, cnum // 2, 3),
            GatedConv2d(cnum // 2, 1, 3, activation=None)
        )

    def forward(self, img, mask):
        coarse_net_input = torch.cat([img * (1 - mask), mask], dims=1)
        coarse_net_output = self.stage_1(coarse_net_input)

        refine_net_input = torch.cat([img * (1 - mask) + coarse_net_output * mask, mask], dims=1)
        refine_net_output = self.stage_2(refine_net_input)

        return coarse_net_output, refine_net_output





'''
Spectral Normalized Patch-GAN Discriminator.
Uses same parameters as original implementation: https://github.com/JiahuiYu/generative_inpainting
Uses SpectralNorm wrapper from: https://github.com/zhaoyuzhi/deepfillv2
'''
class SnPatchGanDiscriminator(nn.Module):
    def __init__(self, base_latent_channels=64):
        super(SnPatchGanDiscriminator, self).__init__()

        # Base channel count
        cnum = base_latent_channels

        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                in_channels=2, out_channels=cnum,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum, out_channels=cnum * 2,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 2, out_channels=cnum * 4,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 4, out_channels=cnum * 4,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 4, out_channels=cnum * 4,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(
                in_channels=cnum * 4, out_channels=cnum * 4,
                kernel_size=5, stride=2, padding=2
            )),
            nn.LeakyReLU(),
            nn.Flatten()
        )

    def forward(self, img, mask):
        net_input = torch.cat([img, mask], dims=1)
        out = self.layers(net_input)
        return out