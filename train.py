import rasterio
import numpy as np
import tqdm
import sys
import time

import torch
import torch.nn as nn

from torch.utils import tensorboard

import models.model_registry
from data.assemble_dataset import DataLoader

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MID_EPOCH_LOG_PERIOD = 10 * 60 # Time in seconds between mid-epoch logs



def train(model_name, epochs=50, print_model_summary=True):
    model, dataloader = models.model_registry.get_model(model_name)

    # Print model summary
    if print_model_summary:
        print(model.summary())

    # Load weights from any previous run
    model.load_weights()

    # Load data
    dataloader.get_data()
    data = dataloader.data
    np.random.shuffle(data)

    # Split data
    val_size = min(int(0.1 * data.shape[0]), 200)
    train_data = data[val_size:]
    val_data = data[:val_size]

    # Begin training epochs
    print(f"\n\nTraining {model.name} (run {model.iteration})")
    for epoch in tqdm.trange(epochs, desc=f"{epochs} Training Epochs"):
        # Time for mid-epoch logging
        last_mid_epoch_log_time = time.time()

        # Reshuffle data
        np.random.shuffle(train_data)

        batch_start = 0
        pbar = tqdm.tqdm(total=(train_data.shape[0] // BATCH_SIZE) * BATCH_SIZE, leave=False)

        # Leave out final incomplete batch
        while batch_start + BATCH_SIZE < train_data.shape[0]:
            # Collect minibatch data
            current_minibatch_size = min(BATCH_SIZE, train_data.shape[0] - batch_start)
            minibatch = train_data[batch_start : batch_start + current_minibatch_size]

            # Run training step
            model.train_minibatch(minibatch)

            # Possible mid-epoch logging
            if time.time() - last_mid_epoch_log_time > MID_EPOCH_LOG_PERIOD:
                model.log_metrics(epoch + (batch_start + current_minibatch_size) / train_data.shape[0], val_data, epoch_complete=False)
                last_mid_epoch_log_time = time.time()

            # Increment and display progress info
            pbar.update(current_minibatch_size)
            pbar.set_description(model.status())
            batch_start += BATCH_SIZE

        # Save model
        model.save_weights()

        # Validate at the end of the epoch
        model.log_metrics(epoch + 1, val_data)



if __name__ == '__main__':
    DEFAULT_NAME = "simple_conv_1"

    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        name = DEFAULT_NAME

    train(name)