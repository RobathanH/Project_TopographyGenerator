import rasterio
import numpy as np
import argparse
from enum import Enum

import torch
import torch.nn as nn

from torch.utils import tensorboard

from models.model_registry import get_model
from models.model_handlers.outpaint_handler import OutPaintHandler
import models.architectures.outpaint_large_noisy_adain as m
from data.get_data import *
from data.util import save_image_lists


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Uses an existing outpaint_large_ada_2 model and generates a novel image from
a randomly or specifically created latent representation.
'''

# Constants
LATENT_WIDTH = 4
LATENT_CHANNELS = 256

# Types of latents to generate
class LatentType(Enum):
    random = "random"
    zero = "zero"
    ones = "ones"
    one_channel_constant = "one_channel_constant"
    one_channel_noise = "one_channel_noise"
    channel_noise = "channel_noise"
    spatial_noise = "spatial_noise"
    channel_spatial_noise_sum = "channel_spatial_noise_sum"

    def __str__(self):
        return self.name


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate a novel heightmap window from scratch")
    argparser.add_argument("latent_type", type=LatentType, choices=list(LatentType), help="Type of latent to generate from")
    argparser.add_argument("-c", "--target_channel", type=int, default=0, help="Channel index for latent types which target a single channel")
    argparser.add_argument("-n", "--image_count", type=int, default=10, help="Number of images to generate")
    args = argparser.parse_args()

    model_handler, dataloader = get_model("outpaint_large_ada_2")
    model_handler.load_weights()
    generator = model_handler.gen

    # Create latent tensors
    if args.latent_type is LatentType.random:
        name = args.latent_type.name
        latents = torch.rand((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.zero:
        name = args.latent_type.name
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.ones:
        name = args.latent_type.name
        latents = torch.ones((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.one_channel_constant:
        name = f"{args.latent_type.name}_{args.target_channel}"
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)
        latents[:, args.target_channel] = torch.ones((args.image_count, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.one_channel_noise:
        name = f"{args.latent_type.name}_{args.target_channel}"
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)
        latents[:, args.target_channel] = torch.rand((args.image_count, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.channel_noise:
        name = args.latent_type.name
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)
        latents += torch.rand((args.image_count, LATENT_CHANNELS, 1, 1)).to(DEVICE)

    elif args.latent_type is LatentType.spatial_noise:
        name = args.latent_type.name
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)
        latents += torch.rand((args.image_count, 1, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)

    elif args.latent_type is LatentType.channel_spatial_noise_sum:
        name = args.latent_type.name
        latents = torch.zeros((args.image_count, LATENT_CHANNELS, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)
        latents += torch.rand((args.image_count, LATENT_CHANNELS, 1, 1)).to(DEVICE)
        latents += torch.rand((args.image_count, 1, LATENT_WIDTH, LATENT_WIDTH)).to(DEVICE)


    # Generate images
    images = np.squeeze(generator.generate_from_latent(latents).detach().cpu().numpy())
    images = [image for image in images] # convert each image to a list element of its own
    save_image_lists([images], main_title="Images from Latents", filename=f"generation_from_latent/{name}.png")