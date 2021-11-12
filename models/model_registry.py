from data.assemble_dataset import DataLoader
from .model_handlers.vae_gan_handler import *
from .model_handlers.deep_fill_handler import *
from .model_handlers.outpaint_handler import *
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

    if name == "simple_deepfill_1":
        # Version of deepfill v2 (without contextual attention 2-branch refinement generator stage)
        # Slightly decreased latent channel count compared to github version
        # Much larger dataset, using larger images (128x128, 20x20 degrees)
        # Dataset filtered to more varied instances
        # Results: Works, but very slow for each epoch and only creates very blurry reconstructions
        import models.architectures.deepfill as m
        model = DeepFillHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.SingleBranchGenerator(32), m.SnPatchGanDiscriminator(32))

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        dataloader.FILTER_ENABLED = True
        dataloader.TRANSLATION_RELATIVE_DISTANCE = 0.05
        
        return model, dataloader

    if name == "simple_deepfill_2":
        # Use default latent channel base counts as original paper,
        # and allow moving square mask
        # RESULTS: WAY TO BIG (14 million params, even though orig paper says they only have 4 million)
        import models.architectures.deepfill as m
        model = DeepFillHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.SingleBranchGenerator(), m.SnPatchGanDiscriminator())
        model.MASK_TYPE = "shifting_inner_square"

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        dataloader.FILTER_ENABLED = True
        dataloader.TRANSLATION_RELATIVE_DISTANCE = 0.05
        
        return model, dataloader

    if name == "simple_deepfill_3":
        # Lower base latent channel count even further than simple_deepfill_1
        # Add learnable instance-norm after every gated conv layer
        # Use shifting square mask
        # Add more disc training iterations since disc loss was stuck at 1 in prev models
        import models.architectures.deepfill as m
        model = DeepFillHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.SingleBranchGenerator(base_latent_channels=16, normalization="in"), m.SnPatchGanDiscriminator(base_latent_channels=24))
        model.MASK_TYPE = "shifting_inner_square"
        model.DISCRIMINATOR_TRAINING_MULTIPLIER = 2

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        dataloader.FILTER_ENABLED = True
        dataloader.TRANSLATION_RELATIVE_DISTANCE = 0.05

        return model, dataloader

    if name == "outpaint_small_1":
        import models.architectures.outpaint_small as m
        model = OutPaintHandler(name, m.IMG_DIMS, m.REGION_DIMS, m.Generator(), m.SnPatchGanDiscriminator(), m.SnPatchGanDiscriminator())

        dataloader = DataLoader(m.IMG_DIMS, m.REGION_DIMS)
        dataloader.FILTER_ENABLED = True

        return model, dataloader


    raise ValueError(f"Unrecognized Name: {name}")