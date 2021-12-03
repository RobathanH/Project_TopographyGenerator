import rasterio
import numpy as np
import tqdm
import sys
import time
import argparse

import torch
import torch.nn as nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.utils import tensorboard

from models.model_registry import get_model
from models.model_handlers.outpaint_handler import OutPaintHandler
from data.get_data import *
from data.util import save_image_lists


# Constants
WINDOW_LNG_LEN = 20
WINDOW_LAT_LEN = 20
WINDOW_X_LEN = 128
WINDOW_Y_LEN = 128

MODEL_NAME_1D = "outpaint_large_ada_2"
MODEL_NAME_2D = "outpaint_large_2d_ada_1"

'''
Repeatedly expand the given seed images in the +x direction using
the given 1D generator. Returns the newly generated windows for each seed,
including the original seed images.
'''
def generate_1d(generator_1d, seed_images, generation_steps):
    test_count = len(seed_images)

    with torch.no_grad():
        expanded_images = torch.zeros(test_count, 1, WINDOW_X_LEN * (1 + generation_steps), WINDOW_Y_LEN, device=DEVICE)

        net_input = torch.cat([
            torch.from_numpy(seed_images).reshape(test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN).to(DEVICE),
            torch.zeros(test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN, device=DEVICE)
        ], dim=2)
        for step in range(args.generation_steps + 1):
            # Generate from net input
            net_output = generator_1d(net_input)

            # Save output as average of generated window and reconstructed window from following generation pass
            expanded_images[:, :, WINDOW_X_LEN * step : WINDOW_X_LEN * (step + 1), :] = (net_input[:, :, :WINDOW_X_LEN, :] + net_output[:, :, :WINDOW_X_LEN, :]) / 2

            # Increment to next net input
            net_input = torch.cat([
                net_output[:, :, WINDOW_X_LEN:, :],
                torch.zeros(test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN, device=DEVICE)
            ], dim=2)

    expanded_images = expanded_images[:, 0, :, :].cpu().numpy()
    return expanded_images

'''
Repeatedly fill the windows between the given 1D image extensions,
using the given 2D generator. Returns the newly generated windows,
not including the given source windows.
'''
def generate_2d(generator_2d, seed_images, pos_x_generation, pos_y_generation):
    test_count = len(seed_images)
    x_generation_steps = pos_x_generation.shape[1] // WINDOW_X_LEN
    y_generation_steps = pos_y_generation.shape[2] // WINDOW_Y_LEN

    with torch.no_grad():
        expanded_images = torch.zeros(test_count, 1, WINDOW_X_LEN * (1 + x_generation_steps), WINDOW_Y_LEN * (1 + y_generation_steps), device=DEVICE)
        expanded_images[:, 0, :WINDOW_X_LEN, :WINDOW_Y_LEN] = torch.from_numpy(seed_images).to(DEVICE)
        expanded_images[:, 0, WINDOW_X_LEN:, :WINDOW_Y_LEN] = torch.from_numpy(pos_x_generation).to(DEVICE)
        expanded_images[:, 0, :WINDOW_X_LEN, WINDOW_Y_LEN:] = torch.from_numpy(pos_y_generation).to(DEVICE)

        for x_step in range(0, x_generation_steps):
            for y_step in range(0, y_generation_steps):
                net_output = generator_2d(expanded_images[:, :, WINDOW_X_LEN * x_step : WINDOW_X_LEN * (x_step + 2), WINDOW_Y_LEN * y_step : WINDOW_Y_LEN * (y_step + 2)])
                expanded_images[:, :, WINDOW_X_LEN * (x_step + 1) : WINDOW_X_LEN * (x_step + 2), WINDOW_Y_LEN * (y_step + 1) : WINDOW_Y_LEN * (y_step + 2)] = net_output[:, :, WINDOW_X_LEN:, WINDOW_Y_LEN:]



    generated_images = expanded_images[:, 0, WINDOW_X_LEN:, WINDOW_Y_LEN:].cpu().numpy()
    return generated_images

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate 2D landmass maps from randomly chosen real map windows")
    argparser.add_argument("-n", "--test_count", type=int, default=1)
    argparser.add_argument("-gx", "--generation_steps_x", type=int, default=2)
    argparser.add_argument("-gy", "--generation_steps_y", type=int, default=2)
    argparser.add_argument("--generator_1d", type=str, default="outpaint_large_4")
    argparser.add_argument("--generator_2d", type=str, default="outpaint_large_2d_2")
    args = argparser.parse_args()

    # Load 1D and 2D models
    model_handler_1D, dataloader = get_model(args.generator_1d)
    model_handler_1D.load_weights()
    generator_1d = model_handler_1D.gen

    model_handler_2D, dataloader = get_model(args.generator_2d)
    model_handler_2D.load_weights()
    generator_2d = model_handler_2D.gen

    # Load seed images
    random_lng_starts = np.random.uniform(LNG_MIN, LNG_MAX, args.test_count)
    random_lat_starts = np.random.uniform(LAT_MIN, LAT_MAX - WINDOW_LAT_LEN, args.test_count)
    seed_images = np.array([
        load_region(random_lng_starts[i], random_lat_starts[i], WINDOW_LNG_LEN, WINDOW_LAT_LEN, output_shape=(WINDOW_X_LEN, WINDOW_Y_LEN))
        for i in range(args.test_count)
    ])

    # Save both generated and reconstructed (when used as source for later generation steps) versions of images,
    # to aid smoothing
    with torch.no_grad():
        # Sum all source, generated, and reconstructed image windows, as well as counts, for averaging at the end
        summed_images = torch.zeros(args.test_count, 1, WINDOW_X_LEN * (1 + args.generation_steps_x), WINDOW_Y_LEN * (1 + args.generation_steps_y), device=DEVICE)
        summed_image_counts = torch.zeros(1 + args.generation_steps_x, 1 + args.generation_steps_y, device=DEVICE)
        
        # Add seed images to sum
        summed_images[:, 0, :WINDOW_X_LEN, :WINDOW_Y_LEN] += torch.from_numpy(seed_images).to(DEVICE)
        summed_image_counts[0, 0] += 1

        # Apply 1D generation in +x direction
        net_input = torch.cat([
            torch.from_numpy(seed_images).reshape(args.test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN).to(DEVICE),
            torch.zeros(args.test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN, device=DEVICE)
        ], dim=2)
        for step in range(args.generation_steps_x):
            net_output = generator_1d(net_input)

            # Add ouput (reconstructed version of source image, and newly generated adjacent image) to sum
            summed_images[:, :, WINDOW_X_LEN * step : WINDOW_X_LEN * (step + 2), :WINDOW_Y_LEN] += net_output
            summed_image_counts[step:(step + 2), 0] += 1

            # Use newly generated image window as next source image window
            net_input = torch.cat([
                net_output[:, :, WINDOW_X_LEN:, :],
                torch.zeros(args.test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN, device=DEVICE)
            ], dim=2)

        # Apply 1D generation in +y direction
        # Requires rotating input image and output images, since 1D generation is always to the right
        net_input = torch.cat([
            torch.from_numpy(seed_images).reshape(args.test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN).rot90(-1, (2, 3)).to(DEVICE),
            torch.zeros(args.test_count, 1, WINDOW_X_LEN, WINDOW_Y_LEN, device=DEVICE)
        ], dim=2)
        for step in range(args.generation_steps_y):
            net_output = generator_1d(net_input)

            # Add ouput (reconstructed version of source image, and newly generated adjacent image) to sum
            summed_images[:, :, :WINDOW_X_LEN, WINDOW_Y_LEN * step : WINDOW_Y_LEN * (step + 2)] += net_output.rot90(1, (2, 3))
            summed_image_counts[0, step:(step + 2)] += 1

            # Use newly generated image window as next source image window
            net_input = torch.cat([
                net_output[:, :, WINDOW_Y_LEN:, :],
                torch.zeros(args.test_count, 1, WINDOW_Y_LEN, WINDOW_X_LEN, device=DEVICE)
            ], dim=2)

        # Apply 2D generation to fill in area between +x and +y generated axes
        for y_step in range(args.generation_steps_y):
            for x_step in range(args.generation_steps_x):
                # Each iteration generates window index x_step + 1, y_step + 1, based
                # on lower three adjacent windows
                
                # Load network input by current average values based on summed images so far
                target_summed_images = summed_images[:, :, WINDOW_X_LEN * x_step : WINDOW_X_LEN * (x_step + 2), WINDOW_Y_LEN * y_step : WINDOW_Y_LEN * (y_step + 2)]
                target_summed_image_counts = summed_image_counts[x_step : x_step + 2, y_step : y_step + 2].reshape(1, 1, 2, 2).repeat_interleave(WINDOW_X_LEN, dim=2).repeat_interleave(WINDOW_Y_LEN, dim=3)
                target_average_image = target_summed_images / target_summed_image_counts
                net_input = target_average_image

                # Pass through 2d generator
                net_output = generator_2d(net_input)

                # Add reconstructed and newly generated images to image sums
                summed_images[:, :, WINDOW_X_LEN * x_step : WINDOW_X_LEN * (x_step + 2), WINDOW_Y_LEN * y_step : WINDOW_Y_LEN * (y_step + 2)] += net_output
                summed_image_counts[x_step : x_step + 2, y_step : y_step + 2] += 1

        # Average all summed images to get final smoothed image
        full_images = summed_images / summed_image_counts.reshape(1, 1, *summed_image_counts.shape).repeat_interleave(WINDOW_X_LEN, dim=2).repeat_interleave(WINDOW_Y_LEN, dim=3)
        full_images = full_images.cpu().numpy().reshape(args.test_count, WINDOW_X_LEN * (1 + args.generation_steps_x), WINDOW_Y_LEN * (1 + args.generation_steps_y))

    # Save resulting images
    images_by_col_row = [list(full_images)]
    save_image_lists(images_by_col_row, filename=f"generation_2d/{args.generation_steps_x}_{args.generation_steps_y}_steps.{args.test_count}_tests.{args.generator_1d}.{args.generator_2d}.png")