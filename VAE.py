import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_DIMS = (128, 128)
LATENT_DIMS = (4, 4)
LATENT_CHANNELS = 16


# Helper Blocks
def ConvPoolBlock(input_channels, output_channels, kernel_size, pool_factor):
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.Tanh(),
        nn.Conv2d(
            in_channels=output_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.MaxPool2d(
            kernel_size=pool_factor, stride=pool_factor, padding=0
        ),
        nn.Tanh()
    )

def ReverseConvPoolBlock(input_channels, output_channels, kernel_size, size_factor, final_activation=True):
    padding = (kernel_size - 1) // 2

    layers = [
        nn.Upsample(
            scale_factor=size_factor, mode='bilinear', align_corners=True
        ),
        nn.Conv2d(
            in_channels=input_channels, out_channels=input_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ),
        nn.Tanh(),
        nn.Conv2d(
            in_channels, input_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
    ]
    if final_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )
        )

    def forward(self, x):
        z = self.layers(x)
        out = nn.ReLU()(x + z)
        return out

# ASSUMES IMG DIMS AND LATENT DIMS HAVE POWER OF 2 RELATIONSHIP,
# SAME EXPONENTIAL FACTOR FOR BOTH DIMS
def ResNetImgDownsize():
    START_CHANNELS = 32
    
    layers = [
        nn.Conv2d(
            in_channels=1, out_channels=START_CHANNELS,
            kernel_size=3, stride=2, padding=1
        ),
        nn.Tanh(),
        ResBlock(START_CHANNELS, 3),
        nn.Conv2d(
            in_channels=START_CHANNELS, out_channels=START_CHANNELS * 2,
            kernel_size=5, stride=4, padding=2
        ),
        nn.Tanh(),
        ResBlock(START_CHANNELS * 2, 5),
        nn.Conv2d(
            in_channels=START_CHANNELS * 2, out_channels=START_CHANNELS * 4,
            kernel_size=5, stride=4, padding=2
        ),
        nn.Tanh()
    ]

    return nn.Sequential(*layers)

class NoisyAdaIn(nn.Module):
    def __init__(self, input_shape, channels):
        super(NoisyAdaIn, self).__init__()

        self.input_shape = input_shape
        self.channels = channels

        self.noise_scale = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.to_mean = nn.Conv2d(
            in_channels=LATENT_CHANNELS, out_channels=channels,
            kernel_size=5, stride=1, padding=2
        )
        self.to_scale = nn.Conv2d(
            in_channels=LATENT_CHANNELS, out_channels=channels,
            kernel_size=5, stride=1, padding=2
        )

    def forward(self, x):
        # Unpack inputs
        x, z = x

        # Add noise
        noise1 = self.noise_scale * torch.randn(1, self.channels, *self.input_shape).to(DEVICE)
        x = x + noise1
        
        # Normalize
        x = nn.InstanceNorm2d(self.channels)(x)

        # Find new per-channel mean and std from latent state
        mean = nn.Upsample(self.input_shape, mode='bilinear', align_corners=True)(self.to_mean(z))
        scale = nn.Upsample(self.input_shape, mode='bilinear', align_corners=True)(self.to_scale(z))

        x = mean + scale * x

        # Nonlinearity
        x = nn.Tanh()(x)

        return x, z


class StyleGanBlock(nn.Module):
    def __init__(self, input_shape, input_channels, hidden_channels, output_channels, kernel_size):
        super(StyleGanBlock, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
        self.ada1 = NoisyAdaIn(input_shape, hidden_channels)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=output_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        )
        self.ada2 = NoisyAdaIn(input_shape, output_channels)

    def forward(self, x):
        # Unpack inputs
        x, z = x

        x = self.conv1(x)
        x, _ = self.ada1((x, z))
        x = self.conv2(x)
        x, _ = self.ada2((x, z))

        return x, z

# Allows a one-input layer to take two inputs and discard the second
class MultiInputWrapper(nn.Module):
    def __init__(self, layer):
        super(MultiInputWrapper, self).__init__()
        self.layer = layer
    
    def forward(self, x):
        # Unpack inputs
        x, z = x[0], x[1:]
        
        return self.layer(x), *z




# Encoder
class VariationalEncoder(nn.Module):
    def __init__(self, architecture=1):
        super(VariationalEncoder, self).__init__()
        self.architecture = architecture

        if self.architecture == 1:
            self.layers = nn.Sequential(
                ConvPoolBlock(1, 32, 3, 2),
                ConvPoolBlock(32, 64, 3, 2),
                ConvPoolBlock(64, 128, 5, 2),
                ConvPoolBlock(128, 256, 5, 2),
                ConvPoolBlock(256, 256, 5, 2),
                ConvPoolBlock(256, 256, 5, 2)
            )

            self.to_mean = nn.Conv2d(
                in_channels=256, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )
            self.to_log_var = nn.Conv2d(
                in_channels=256, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )
        
        elif self.architecture in [2, 5]:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=32,
                    kernel_size=3, stride=2, padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=32, out_channels=64,
                    kernel_size=3, stride=2, padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=64, out_channels=128,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=128, out_channels=256,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh()
            )

            self.to_mean = nn.Conv2d(
                in_channels=256, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )
            self.to_log_var = nn.Conv2d(
                in_channels=256, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )

        elif self.architecture in [3, 4]:
            self.layers = ResNetImgDownsize()
            self.to_mean = nn.Conv2d(
                in_channels=128, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )
            self.to_log_var = nn.Conv2d(
                in_channels=128, out_channels=LATENT_CHANNELS,
                kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        if self.architecture in [1,2,3,4,5]:
            x = self.layers(x)
            z_mean = self.to_mean(x)
            z_log_var = self.to_log_var(x)
            return z_mean, z_log_var

        else:
            return None

# Sampler
class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()
    
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + (eps * std)
        return z


# Decoder
class Decoder(nn.Module):
    def __init__(self, architecture=1):
        super(Decoder, self).__init__()
        self.architecture = architecture

        if self.architecture == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=LATENT_CHANNELS, out_channels=256,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Tanh(),
                ReverseConvPoolBlock(256, 256, 5, 2),
                ReverseConvPoolBlock(256, 256, 5, 2),
                ReverseConvPoolBlock(256, 128, 5, 2),
                ReverseConvPoolBlock(128, 64, 5, 2),
                ReverseConvPoolBlock(64, 32, 3, 2),
                ReverseConvPoolBlock(32, 1, 3, 2, final_activation=False)
            )

        elif self.architecture == 2:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=LATENT_CHANNELS, out_channels=256,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Tanh(),
                nn.ConvTranspose2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=1, padding=2
                ),
                nn.Tanh(),
                nn.ConvTranspose2d(
                    in_channels=256, out_channels=128,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=128, out_channels=128,
                    kernel_size=5, stride=1, padding=2
                ),
                nn.Tanh(),
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=64, out_channels=64,
                    kernel_size=5, stride=1, padding=2
                ),
                nn.Tanh(),
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=32, out_channels=32,
                    kernel_size=3, stride=1, padding=1
                ),
                nn.Tanh(),
                nn.ConvTranspose2d(
                    in_channels=32, out_channels=16,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=16, out_channels=1,
                    kernel_size=3, stride=1, padding=1
                )
            )

        elif self.architecture == 3:
            self.layers = nn.Sequential(
                MultiInputWrapper(nn.Conv2d(
                    in_channels=LATENT_CHANNELS, out_channels=32,
                    kernel_size=1, stride=1, padding=0
                )),
                StyleGanBlock((4, 4), 32, 64, 64, 3),
                MultiInputWrapper(nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                )),
                StyleGanBlock((8, 8), 64, 128, 128, 3),
                MultiInputWrapper(nn.Upsample(
                    scale_factor=4, mode='bilinear', align_corners=True
                )),
                StyleGanBlock((32, 32), 128, 256, 256, 5),
                MultiInputWrapper(nn.Upsample(
                    scale_factor=4, mode='bilinear', align_corners=True
                )),
                StyleGanBlock((128, 128), 256, 128, 64, 5),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=64, out_channels=1,
                    kernel_size=1, stride=1, padding=0
                ))
            )

        elif self.architecture == 4:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=LATENT_CHANNELS, out_channels=32,
                    kernel_size=3, stride=1, padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=32, out_channels=32,
                    kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(32, affine=True),
                nn.Tanh(),
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ),
                nn.Conv2d(
                    in_channels=32, out_channels=64,
                    kernel_size=3, stride=1, dilation=2, padding=2
                ),
                nn.InstanceNorm2d(64, affine=True),
                nn.Tanh(),
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ),
                nn.Conv2d(
                    in_channels=64, out_channels=128,
                    kernel_size=5, stride=1, dilation=2, padding=4
                ),
                nn.InstanceNorm2d(128, affine=True),
                nn.Tanh(),
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ),
                nn.Conv2d(
                    in_channels=128, out_channels=256,
                    kernel_size=5, stride=1, dilation=2, padding=4
                ),
                nn.InstanceNorm2d(256, affine=True),
                nn.Tanh(),
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ),
                nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=1, dilation=2, padding=4
                ),
                nn.InstanceNorm2d(256, affine=True),
                nn.Tanh(),
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ),
                nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=1, dilation=2, padding=4
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=256, out_channels=1,
                    kernel_size=5, stride=1, padding=2
                )
            )

        elif self.architecture == 5:
            self.layers = nn.Sequential(
                MultiInputWrapper(nn.Conv2d(
                    in_channels=LATENT_CHANNELS, out_channels=256,
                    kernel_size=1, stride=1, padding=0
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.ConvTranspose2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=1, padding=2
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.ConvTranspose2d(
                    in_channels=256, out_channels=128,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=128, out_channels=128,
                    kernel_size=5, stride=1, padding=2
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=64, out_channels=64,
                    kernel_size=5, stride=1, padding=2
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=32, out_channels=32,
                    kernel_size=3, stride=1, padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                MultiInputWrapper(nn.ConvTranspose2d(
                    in_channels=32, out_channels=32,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )),
                MultiInputWrapper(nn.Tanh()),
                StyleGanBlock((128, 128), 32, 32, 32, 5),
                MultiInputWrapper(nn.Conv2d(
                    in_channels=32, out_channels=1,
                    kernel_size=3, stride=1, padding=1
                ))
            )

    def forward(self, x):
        if self.architecture in [1, 2, 4]:
            return self.layers(x)

        elif self.architecture in [3, 5]:
            z = x
            return self.layers((x, z))[0]

        else:
            return None


class ConvVAE(nn.Module):
    def __init__(self, architecture=1):
        super(ConvVAE, self).__init__()
        self.architecture = architecture

        self.encoder = VariationalEncoder(architecture)
        self.sampler = Sampler()
        self.decoder = Decoder(architecture)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

class Discriminator(nn.Module):
    def __init__(self, architecture=1):
        super(Discriminator, self).__init__()
        self.architecture = architecture

        if self.architecture in [1, 2, 5]:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=32,
                    kernel_size=3, stride=2, padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=32, out_channels=64,
                    kernel_size=3, stride=2, padding=1
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=64, out_channels=128,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=128, out_channels=256,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=256, out_channels=256,
                    kernel_size=5, stride=2, padding=2
                ),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=256, out_channels=16,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(in_features=256, out_features=1),
                nn.Sigmoid()
            )

        if self.architecture in [3, 4]:
            self.layers = nn.Sequential(
                ResNetImgDownsize(),
                nn.Conv2d(
                    in_channels=128, out_channels=16,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Flatten(),
                nn.Linear(in_features=256, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.architecture in [1, 2, 3, 4, 5]:
            return self.layers(x)

        else:
            return None


def train_vae_gan(vae, discriminator, data, epochs=20, vae_savename=None, discriminator_savename=None):
    # Constants
    BATCH_SIZE = 32
    RECON_RELATIVE_LOSS_WEIGHT = 1
    KL_RELATIVE_LOSS_WEIGHT = 1
    GAN_RELATIVE_LOSS_WEIGHT = 0.1
    DISCRIMINATOR_TRAINING_MULTIPLIER = 1 # Discriminator can do more updates for each vae update

    # Compute average input height variance
    data_average_var = np.mean(np.var(data, axis=(1,2)))

    vae_opt = torch.optim.Adam(vae.parameters())
    discriminator_opt = torch.optim.Adam(discriminator.parameters())

    vae_loss_over_epochs = []
    discriminator_loss_over_epochs = []
    for epoch in range(epochs):
        vae_epoch_loss = 0.0
        discriminator_epoch_loss = 0.0
        vae_loss_within_epoch = []
        discriminator_loss_within_epoch = []

        # average epoch losses for parts of vae loss (after weight multipliers)
        recon_epoch_loss = 0.0
        kl_epoch_loss = 0.0
        gan_epoch_loss = 0.0

        # Reshuffle data
        np.random.shuffle(data)

        batch_start = 0
        pbar = tqdm.tqdm(total=data.shape[0])
        while batch_start < data.shape[0]:

            # Prepare minibatch data
            current_minibatch_size = min(BATCH_SIZE, data.shape[0] - batch_start)
            x = data[batch_start : batch_start + current_minibatch_size]
            x = x.reshape(x.shape[:1] + (1,) + x.shape[1:])
            x = torch.tensor(x).to(DEVICE)

            # Save variance in each input sample to scale reconstruction loss
            #x_var = torch.var(x, dim=(1,2,3), keepdim=True)


            # Discriminator Update
            for _ in range(DISCRIMINATOR_TRAINING_MULTIPLIER):
                discriminator_opt.zero_grad()
                reconstruction, z_mean, z_log_var = vae(x)
                discriminator_loss = (torch.sum(-torch.log(discriminator(x))) + torch.sum(-torch.log(1 - discriminator(reconstruction)))) / 2
                discriminator_loss.backward()
                discriminator_opt.step()

            # VAE Update
            vae_opt.zero_grad()
            reconstruction, z_mean, z_log_var = vae(x)
            recon_loss = RECON_RELATIVE_LOSS_WEIGHT * torch.sum(torch.mean(((x - reconstruction).pow(2) / (2 * data_average_var)), axis=(1, 2, 3)))
            kl_loss = KL_RELATIVE_LOSS_WEIGHT * -0.5 * torch.sum(torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), axis=1))
            gan_loss = GAN_RELATIVE_LOSS_WEIGHT * -torch.sum(torch.log(discriminator(reconstruction)))
            vae_loss = (recon_loss + kl_loss) + gan_loss
            vae_loss.backward()
            vae_opt.step()

            # Save losses
            vae_epoch_loss += vae_loss.item() / data.shape[0]
            discriminator_epoch_loss += discriminator_loss.item() / data.shape[0]
            vae_loss_within_epoch.append(vae_loss.item())
            discriminator_loss_within_epoch.append(discriminator_loss.item())

            recon_epoch_loss += recon_loss.item() / data.shape[0]
            kl_epoch_loss += kl_loss.item() / data.shape[0]
            gan_epoch_loss += gan_loss.item() / data.shape[0]

            # Increment and display info
            running_average_correction = data.shape[0] / (batch_start + current_minibatch_size)
            vae_running_loss = vae_epoch_loss * running_average_correction
            recon_running_loss = recon_epoch_loss * running_average_correction
            kl_running_loss = kl_epoch_loss * running_average_correction
            gan_running_loss = gan_epoch_loss * running_average_correction
            discriminator_running_loss = discriminator_epoch_loss * running_average_correction

            pbar.update(min(BATCH_SIZE, data.shape[0] - batch_start))
            pbar.set_description(f"Epoch {epoch}, VAE Loss = {vae_running_loss:.3f} ({recon_running_loss:.3f}, {kl_running_loss:.3f}, {gan_running_loss:.3f}), Disc Loss = {discriminator_running_loss:.4f}")
            batch_start += BATCH_SIZE

        # Save loss
        vae_loss_over_epochs.append(vae_epoch_loss)
        discriminator_loss_over_epochs.append(discriminator_epoch_loss)

        # Save Models
        if vae_savename is not None:
            torch.save(vae.state_dict(), vae_savename)
        if discriminator_savename is not None:
            torch.save(discriminator.state_dict(), discriminator_savename)

        # Debug
        if True:
            show(vae, data[0:10], f"logs/training_example_arch3_1.{epoch}.png")

    # Debug
    if True:
        fig, axarr = plt.subplots(1, 2)
        axarr[0].plot(vae_loss_over_epochs)
        axarr[0].title.set_text("Vae Loss Over Epochs")
        axarr[1].plot(discriminator_loss_over_epochs)
        axarr[1].title.set_text("Discriminator Loss Over Epochs")
        plt.savefig('training_loss.png')
        plt.show()
        plt.close('all')

def train_vae(vae, data, epochs=20, savename=None):
    # Constants
    BATCH_SIZE = 32

    # Compute average input height variance
    data_average_var = np.mean(np.var(data, axis=(1,2)))

    opt = torch.optim.Adam(vae.parameters())
    
    loss_over_epochs = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        loss_within_epoch = []

        # Reshuffle data
        np.random.shuffle(data)
        
        batch_start = 0
        pbar = tqdm.tqdm(total=data.shape[0])
        while batch_start < data.shape[0]:
            x = data[batch_start : min(batch_start + BATCH_SIZE, data.shape[0])]
            x = x.reshape(x.shape[:1] + (1,) + x.shape[1:])
            x = torch.tensor(x).to(DEVICE)

            #x_var = torch.var(x, dim=(1,2,3))

            # VAE Update
            opt.zero_grad()
            reconstruction, z_mean, z_log_var = vae(x)
            recon_loss = torch.mean(((x - reconstruction).pow(2) / (2 * data_average_var)))
            kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            opt.step()
            

            # Save loss
            epoch_loss += loss.item() / data.shape[0]
            loss_within_epoch.append(loss.item())


            # Increment and display info
            pbar.update(min(BATCH_SIZE, data.shape[0] - batch_start))
            pbar.set_description(f"Epoch {epoch}, Loss = {epoch_loss * (data.shape[0] / min(batch_start + BATCH_SIZE, data.shape[0])) : .4f}")
            batch_start += BATCH_SIZE

        # Save loss
        loss_over_epochs.append(epoch_loss)

        # Save model
        if savename is not None:
            torch.save(vae.state_dict(), savename)

        # Debug
        if False:
            print(f"Epoch {epoch} Loss: {epoch_loss}")
        if False:
            plt.plot(loss_within_epoch)
            plt.title(f"Loss Within Epoch {epoch}")
            plt.show()
        if True:
            show(vae, data[np.random.choice(data.shape[0])])

    # Debug
    if True:
        plt.plot(loss_over_epochs)
        plt.title("Loss Over Epochs")
        plt.savefig('training_loss.png')
        plt.show()

def show(vae, samples, savepath=None):
    count = samples.shape[0]
    net_input = samples.reshape(samples.shape[:1] + (1,) + samples.shape[1:])
    net_input = torch.tensor(net_input).to(DEVICE)
    net_output, _, _ = vae(net_input)

    reconstructions = net_output.detach().cpu().numpy().reshape(samples.shape)

    fig, axarr = plt.subplots(count, 2, figsize=(12, count * 6))
    for i in range(count):
        fixed_zero_imshow(samples[i], axarr[i][0])
        axarr[i][0].title.set_text("Original")
        fixed_zero_imshow(reconstructions[i], axarr[i][1])
        axarr[i][1].title.set_text("Reconstruction")
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close('all')

def fixed_zero_imshow(img, ax):
    max_mag = np.max(np.abs(img))
    ax.imshow(img, vmin=-max_mag, vmax=max_mag, cmap='coolwarm')


if __name__ == "__main__":
    # Model Constants
    ARCHITECTURE = 4
    VAE_SAVENAME = f"models/arch{ARCHITECTURE}_vae_weights.pth"
    DISCRIMINATOR_SAVENAME = f"models/arch{ARCHITECTURE}_discriminator_weights.pth"

    # Data Constants
    FILTER_DATA = True
    ZOOM_FACTOR = 4
    TRANSLATION_FACTOR = 10
    MAX_DATA_MEM_SIZE = 4 * 2**30 # 4 GB

    def img_filepath(counter):
        return f"data/10x10_windows.{counter}.img.npy"

    if os.path.exists(f"compiled_data{'_filtered' if FILTER_DATA else '_unfiltered'}.npy"):
        data = np.load(f"compiled_data{'_filtered' if FILTER_DATA else '_unfiltered'}.npy")
    else:
        data = None
        counter = 1
        while (data is None or data.size * data.itemsize <= MAX_DATA_MEM_SIZE) and os.path.exists(img_filepath(counter)):
            # Load data
            file_data = np.load(img_filepath(counter))

            # Rescale data (measure in km)
            file_data /= 1000.0

            # Subdivide
            new_x_len = file_data.shape[1] // ZOOM_FACTOR
            new_y_len = file_data.shape[2] // ZOOM_FACTOR
            shift_amount_x = (file_data.shape[1] - new_x_len) // TRANSLATION_FACTOR
            shift_amount_y = (file_data.shape[2] - new_y_len) // TRANSLATION_FACTOR
            file_data = np.concatenate([
                file_data[:, i * shift_amount_x : i * shift_amount_x + new_x_len, j * shift_amount_y : j * shift_amount_y + new_y_len] 
                for i in range(TRANSLATION_FACTOR) 
                for j in range(TRANSLATION_FACTOR)
            ], axis=0)

            # Downsample
            x_inds = np.round(np.linspace(0, file_data.shape[1], num=IMG_DIMS[0], endpoint=False)).astype(int)
            y_inds = np.round(np.linspace(0, file_data.shape[2], num=IMG_DIMS[1], endpoint=False)).astype(int)
            file_data = file_data[:, x_inds, :]
            file_data = file_data[:, :, y_inds]

            # Filter data
            if FILTER_DATA:
                variance = np.var(file_data, axis=(-1, -2))
                fraction_above_water = np.sum(file_data > 0, axis=(-1, -2)) / (file_data.shape[-1] * file_data.shape[-2])
                valid_variance = variance >= 0.1**2
                valid_land_mass = fraction_above_water >= 0.05
                valid = valid_variance * valid_land_mass
                print(f"Samples with valid variance: {np.sum(valid_variance)} / {file_data.shape[0]}")
                print(f"Samples with valid landmass: {np.sum(valid_land_mass)} / {file_data.shape[0]}")
                print(f"Overall valid samples: {np.sum(valid)} / {file_data.shape[0]}")
                file_data = file_data[valid]

            # Add file data to overall data
            if data is None:
                data = file_data
            else:
                data = np.concatenate([data, file_data])

            # Increment counter
            counter += 1

        # Shuffle data
        np.random.shuffle(data)

        # Save data for easy reuse
        np.save(f"compiled_data{'_filtered' if FILTER_DATA else '_unfiltered'}.npy", data)

    # train/val split
    train_data_size = int(data.shape[0] * 0.9)
    train_data = data[:train_data_size]
    val_data = data[train_data_size:]

    # Create VAE
    vae = ConvVAE(architecture=ARCHITECTURE).to(DEVICE)
    print("\nVAE:")
    print(f"{sum([np.prod(p.size()) for p in vae.parameters() if p.requires_grad])} params")
    print(vae)

    # Load VAE
    if True and os.path.exists(VAE_SAVENAME):
        vae.load_state_dict(torch.load(VAE_SAVENAME))
    show(vae, train_data[0:5], "logs/on_load_example.png")

    # Create Discriminator
    discriminator = Discriminator(architecture=ARCHITECTURE).to(DEVICE)
    print("\Discriminator:")
    print(f"{sum([np.prod(p.size()) for p in discriminator.parameters() if p.requires_grad])} params")
    print(discriminator)

    # Load Discriminator
    if True and os.path.exists(DISCRIMINATOR_SAVENAME):
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_SAVENAME))

    

    # Train (and save)
    train_vae_gan(vae, discriminator, train_data, epochs=50, vae_savename=VAE_SAVENAME, discriminator_savename=DISCRIMINATOR_SAVENAME)

    # Show results
    show(vae, val_data[0:10], "logs/final_val_example.png")