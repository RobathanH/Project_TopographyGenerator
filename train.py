from re import split
import numpy as np
import tqdm

import torch
import torch.nn as nn

from torch.utils import tensorboard

import models.model_registry
from data.assemble_dataset import DataLoader

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32



def train(model_name, epochs=20):
    model = models.model_registry.get_model(model_name)

    # Load weights from any previous run
    model.load_weights()

    # Load data
    data = DataLoader(model.img_dims, model.region_dims)
    data = data.load_full_data()
    np.random.shuffle(data)

    # Split data
    split_ind = round(data.shape[0] * 0.9)
    train_data = data[:split_ind]
    val_data = data[split_ind:]

    # Begin training epochs
    for epoch in range(epochs):
        # Reshuffle data
        np.random.shuffle(train_data)

        batch_start = 0
        pbar = tqdm.tqdm(total=train_data.shape[0])
        while batch_start < train_data.shape[0]:
            # Collect minibatch data
            current_minibatch_size = min(BATCH_SIZE, train_data.shape[0] - batch_start)
            minibatch = train_data[batch_start : batch_start + current_minibatch_size]

            # Run training step
            model.train_minibatch(minibatch)

            # Increment and display progress info
            pbar.update(current_minibatch_size)
            pbar.set_description(model.status())
            batch_start += BATCH_SIZE

        # Save model
        model.save_weights()

        # Validate at the end of the epoch
        model.epoch_complete(epoch, val_data)



if __name__ == '__main__':
    train("test")