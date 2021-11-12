from enum import Enum

import torch
import torch.nn as nn

class Normalizer(Enum):
    NONE = 0
    BATCH_NORM = 1
    INSTANCE_NORM = 2
    LEARNABLE_INSTANCE_NORM = 3

    def create_layer(self, in_features):
        if self is Normalizer.NONE:
            return nn.Identity()
        if self is Normalizer.BATCH_NORM:
            return nn.BatchNorm2d(in_features)
        if self is Normalizer.INSTANCE_NORM:
            return nn.InstanceNorm2d(in_features)
        if self is Normalizer.LEARNABLE_INSTANCE_NORM:
            return nn.InstanceNorm2d(in_features, affine=True)