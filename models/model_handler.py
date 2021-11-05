import torch
import torch.nn as nn
from torch.utils import tensorboard
import os
import numpy as np

from .util import *
import data.util

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FOLDER = 'logs'
VAL_BATCH_SIZE = 32



'''
Interface
'''
class ModelHandler:
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

    def log_folder(self):
        return os.path.join(self.model_folder(), f"run_{self.iteration}")

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



class VaeGanHandler(ModelHandler):
    def __init__(self, name, img_dims, region_dims, encoder, decoder, discriminator):
        super(VaeGanHandler, self).__init__(name, img_dims, region_dims)

        # Save networks
        self.vae = ConvVAE(encoder, decoder).to(DEVICE)
        self.discriminator = discriminator.to(DEVICE)

        # Optimizers
        self.vae_opt = torch.optim.Adam(self.vae.parameters())
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters())

        # Training Constants
        self.RECON_RELATIVE_LOSS_WEIGHT = 1
        self.KL_RELATIVE_LOSS_WEIGHT = 1
        self.GAN_RELATIVE_LOSS_WEIGHT = 0.1
        self.DISCRIMINATOR_TRAINING_MULTIPLIER = 1 # Discriminator can do more updates for each vae update

        # State variables - updated over each epoch
        self.epoch_size = 0
        self.epoch_vae_loss = 0
        self.epoch_vae_recon_loss = 0
        self.epoch_vae_kl_loss = 0
        self.epoch_vae_gan_loss = 0
        self.epoch_disc_loss = 0

    def load_weights(self):
        vae_savepath = os.path.join(self.log_folder(), "vae_weights.pth")
        disc_savepath = os.path.join(self.log_folder(), "disc_weights.pth")
        if os.path.exists(vae_savepath) and os.path.exists(disc_savepath):
            self.vae.load_state_dict(torch.load(vae_savepath))
            self.discriminator.load_state_dict(torch.load(disc_savepath))

    def save_weights(self):
        vae_savepath = os.path.join(self.log_folder(), "vae_weights.pth")
        disc_savepath = os.path.join(self.log_folder(), "disc_weights.pth")
        torch.save(self.vae.state_dict(), vae_savepath)
        torch.save(self.discriminator.state_dict(), disc_savepath)

    def status(self):
        message = "VAE Loss = {:.3f} ({:.3f}, {:.3f}, {:.3f}), Discriminator Loss = {:.3f}".format(
            self.epoch_vae_loss / self.epoch_size,
            self.epoch_vae_recon_loss / self.epoch_size,
            self.epoch_vae_kl_loss / self.epoch_size,
            self.epoch_vae_gan_loss / self.epoch_size,
            self.epoch_disc_loss / self.epoch_size
        )
        return message

    def train_minibatch(self, minibatch):
        minibatch = minibatch.reshape(minibatch.shape[:1] + (1,) + minibatch.shape[1:])
        x = torch.tensor(minibatch).to(DEVICE)

        # Discriminator Update
        for _ in range(self.DISCRIMINATOR_TRAINING_MULTIPLIER):
            self.discriminator_opt.zero_grad()
            discriminator_loss = self.discriminator_loss_fn(x)
            discriminator_loss.backward()
            self.discriminator_opt.step()

        # VAE Update
        self.vae_opt.zero_grad()
        recon_loss, kl_loss, gan_loss = self.vae_loss_fn(x)
        vae_loss = recon_loss + kl_loss + gan_loss
        vae_loss.backward()
        self.vae_opt.step()

        # Save metrics
        self.epoch_size += 1
        self.epoch_vae_loss += vae_loss.item()
        self.epoch_vae_recon_loss += recon_loss.item()
        self.epoch_vae_kl_loss += kl_loss.item()
        self.epoch_vae_gan_loss += gan_loss.item()
        self.epoch_disc_loss += discriminator_loss.item()

    def epoch_complete(self, epoch_idx, val_data):
        # Find val loss
        total_val_vae_loss = 0
        total_val_vae_recon_loss = 0
        total_val_vae_kl_loss = 0
        total_val_vae_gan_loss = 0
        total_val_disc_loss = 0

        batch_start = 0
        while batch_start < val_data.shape[0]:
            batch_size = min(VAL_BATCH_SIZE, val_data.shape[0] - batch_start)
            val_batch = val_data[batch_start : batch_start + batch_size]
            val_x = torch.tensor(val_batch.reshape(val_batch.shape[:1] + (1,) + val_batch.shape[1:])).to(DEVICE)
            val_disc_loss = self.discriminator_loss_fn(val_x)
            val_vae_recon_loss, val_vae_kl_loss, val_vae_gan_loss = self.vae_loss_fn(val_x)
            val_vae_loss = val_vae_recon_loss + val_vae_kl_loss + val_vae_gan_loss

            total_val_vae_loss += val_vae_loss.item()
            total_val_vae_recon_loss += val_vae_recon_loss.item()
            total_val_vae_kl_loss += val_vae_kl_loss.item()
            total_val_vae_gan_loss += val_vae_gan_loss.item()
            total_val_disc_loss += val_disc_loss.item()

            batch_start += VAL_BATCH_SIZE
        val_vae_loss = total_val_vae_loss / val_data.shape[0]
        val_vae_recon_loss = total_val_vae_recon_loss / val_data.shape[0]
        val_vae_kl_loss = total_val_vae_kl_loss / val_data.shape[0]
        val_vae_gan_loss = total_val_vae_gan_loss / val_data.shape[0]
        val_disc_loss = total_val_disc_loss / val_data.shape[0]

        # Log val loss
        self.writer.add_scalar("eval/vae_loss", val_vae_loss, epoch_idx)
        self.writer.add_scalar("eval/vae_recon_loss", val_vae_recon_loss, epoch_idx)
        self.writer.add_scalar("eval/vae_kl_loss", val_vae_kl_loss, epoch_idx)
        self.writer.add_scalar("eval/vae_gan_loss", val_vae_gan_loss, epoch_idx)
        self.writer.add_scalar("eval/discriminator_loss", val_disc_loss, epoch_idx)

        # Log train loss
        self.writer.add_scalar("train/vae_loss", self.epoch_vae_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/vae_recon_loss", self.epoch_vae_recon_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/vae_kl_loss", self.epoch_vae_kl_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/vae_gan_loss", self.epoch_vae_gan_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/discriminator_loss", self.epoch_disc_loss / self.epoch_size, epoch_idx)

        # Reset train loss vars
        self.epoch_size = 0
        self.epoch_vae_loss = 0
        self.epoch_vae_recon_loss = 0
        self.epoch_vae_kl_loss = 0
        self.epoch_vae_gan_loss = 0
        self.epoch_disc_loss = 0

        # Save some examples from the val_data
        rand_inds = np.random.choice(np.arange(val_data.shape[0]), 20)
        example_imgs = val_data[rand_inds]
        net_input = torch.tensor(example_imgs.reshape(example_imgs.shape[:1] + (1,) + example_imgs.shape[1:])).to(DEVICE)
        net_output, _, _ = self.vae(net_input)
        example_reconstructions = net_output.detach().cpu().numpy().reshape(example_imgs.shape)
        data.util.save_images2(example_imgs, example_reconstructions, main_title=f"{self.name}_epoch{epoch_idx}",
                                title1="Original", title2="Reconstruction", filename=os.path.join(self.log_folder(), f"epoch_{epoch_idx}.png"))




    # Private Helper Functions
    def discriminator_loss_fn(self, x):
        reconstruction, z_mean, z_log_var = self.vae(x)
        discriminator_loss = (torch.sum(-torch.log(self.discriminator(x))) + torch.sum(-torch.log(1 - self.discriminator(reconstruction)))) / 2
        return discriminator_loss

    def vae_loss_fn(self, x):
        reconstruction, z_mean, z_log_var = self.vae(x)
        recon_loss = self.RECON_RELATIVE_LOSS_WEIGHT * torch.sum(torch.mean(((x - reconstruction).pow(2) / 2), axis=(1, 2, 3)))
        kl_loss = self.KL_RELATIVE_LOSS_WEIGHT * -0.5 * torch.sum(torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), axis=1))
        gan_loss = self.GAN_RELATIVE_LOSS_WEIGHT * -torch.sum(torch.log(self.discriminator(reconstruction)))
        return recon_loss, kl_loss, gan_loss

