from data.assemble_dataset import DataLoader
from .model_handlers.vae_gan_handler import *
from .model_handlers.deep_fill_handler import *
from .util import *


def get_model(name):
    if name == "simple_conv_vae_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 0

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        return model, dataloader

    if name == "simple_conv_vae_gan_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Encoder(), m.Decoder(), m.Discriminator())

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        return model, dataloader

    if name == "simple_conv_gan_1":
        import models.architectures.simple_conv_1 as m
        model = VaeGanHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 1
        model.RECON_RELATIVE_LOSS_WEIGHT = 0
        model.KL_RELATIVE_LOSS_WEIGHT = 0

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        return model, dataloader

    if name == "resnet_conv_vae_gan_1":
        import models.architectures.resnet_conv_1 as m
        model = VaeGanHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Encoder(), m.Decoder(), m.Discriminator())

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        return model, dataloader

    if name == "resnet_conv_vae_1":
        import models.architectures.resnet_conv_1 as m
        model = VaeGanHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Encoder(), m.Decoder(), m.Discriminator())
        model.GAN_RELATIVE_LOSS_WEIGHT = 0

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        return model, dataloader

    if name == "deepfill_1":
        import models.architectures.deepfill as m
        model = DeepFillHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.SingleBranchGenerator(), m.SnPatchGanDiscriminator())

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        dataloader.FILTER_ENABLED = True
        
        return model, dataloader


    raise ValueError(f"Unrecognized Name: {name}")