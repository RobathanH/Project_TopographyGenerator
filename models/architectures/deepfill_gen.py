import torch
import torch.nn as nn
from ..conv_blocks import *

IMG_DIMS = (128, 128)
REGION_DIMS = (10, 10)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Base channel count
        cnum = 32

        # Stage 1
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

        # Stage 2 - conv branch
        self.stage_2_conv = nn.Sequential(
            GatedConv2d(1, cnum, 5),
            GatedConv2d(cnum, cnum, 3, stride=2),
            GatedConv2d(cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 2 * cnum, 3, stride=2),
            GatedConv2d(2 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=4),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=8),
            GatedConv2d(4 * cnum, 4 * cnum, 3, dilation=16)
        )

        # Stage 2 - attention branch
        self.stage_2_attention = nn.Sequential(
            GatedConv2d(1, cnum, 5),
            GatedConv2d(cnum, cnum, 3, stride=2),
            GatedConv2d(cnum, 2 * cnum, 3),
            GatedConv2d(2 * cnum, 4 * cnum, 3, stride=2),
            GatedConv2d(4 * cnum, 4 * cnum, 3),
            GatedConv2d(4 * cnum, 4 * cnum, 3, activation=nn.ReLU())
        )

