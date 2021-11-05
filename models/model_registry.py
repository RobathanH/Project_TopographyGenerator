from .model_handler import *
from .util import *


def get_model(name):
    if name == "simple_conv_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 0
        return model

    else:
        raise ValueError(f"Unrecognized Name: {name}")