from enum import Enum

import torch
import torch.nn as nn

class Activation(Enum):
    NONE = 1
    RELU = 2
    LRELU = 3
    ELU = 4
    SIGMOID = 5
    TANH = 6

    def create_layer(self):
        if self is Activation.NONE:
            return nn.Identity()
        if self is Activation.RELU:
            return nn.ReLU()
        if self is Activation.LRELU:
            return nn.LeakyReLU()
        if self is Activation.ELU:
            return nn.ELU()
        if self is Activation.SIGMOID:
            return nn.Sigmoid()
        if self is Activation.TANH:
            return nn.Tanh()