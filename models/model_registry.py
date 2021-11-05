from .model_handler import *
from .util import *


def get_model(name):
    if name == "simple_conv_vae_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 0
        return model

    if name == "simple_conv_vae_gan_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        return model

    if name == "simple_conv_gan_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 1
        model.RECON_RELATIVE_LOSS_WEIGHT = 0
        model.KL_RELATIVE_LOSS_WEIGHT = 0
        return model

    if name == "resnet_conv_vae_gan_1":
        import models.architectures.resnet_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        return model

    if name == "resnet_conv_vae_1":
        import models.architectures.resnet_conv_1 as m
        model = VaeGanHandler(name, (64, 64), (20, 20), m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 0
        return model


    raise ValueError(f"Unrecognized Name: {name}")