import torch
import torch.nn as nn
from torch.utils import tensorboard
import os
import numpy as np

from ..util import *
import data.util


# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FOLDER = 'logs'
VAL_BATCH_SIZE = 32



'''
Interface
'''
class ModelHandlerBase:
    def __init__(self, name, img_dims, region_dims):
        self.name = name
        self.img_dims = img_dims
        self.region_dims = region_dims

        # Find latest log iteration for this model name
        prev_iteration = 0
        if os.path.exists(self.model_folder()):
            for file in os.listdir(self.model_folder()):
                file = file.split('_')
                if len(file) == 2 and file[0] == "run" and int(file[1]) > prev_iteration:
                    prev_iteration = int(file[1])
        
        self.iteration = prev_iteration + 1

        self.writer = tensorboard.SummaryWriter(log_dir=self.log_folder())


    def model_folder(self):
        script_folder = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(script_folder, LOG_FOLDER, self.name)

    def log_folder(self, iteration=None):
        if iteration is None:
            iteration = self.iteration
        return os.path.join(self.model_folder(), f"run_{iteration}")

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError

    '''
    Args:
        minibatch (np.ndarray): shape = (batch_size,) + img_dims
    '''
    def train_minibatch(self, minibatch):
        raise NotImplementedError

    # Triggers resetting epoch average loss vars, and writing to tensorboard
    def epoch_complete(self, epoch_idx, val_data):
        raise NotImplementedError

    # Triggers short status message about current average loss
    def status(self):
        raise NotImplementedError

    # Returns summaries and param-counts for all contained models
    def summary(self):
        raise NotImplementedError