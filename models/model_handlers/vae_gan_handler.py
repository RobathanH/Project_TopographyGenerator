from .model_handler_base import *

class VaeGanHandler(ModelHandlerBase):
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
        for prev_it in range(1, self.iteration)[::-1]:
            vae_savepath = os.path.join(self.log_folder(prev_it), "vae_weights.pth")
            disc_savepath = os.path.join(self.log_folder(prev_it), "disc_weights.pth")
            if os.path.exists(vae_savepath) and os.path.exists(disc_savepath):
                print(f"Loading weights from {self.name}, run {prev_it}")
                self.vae.load_state_dict(torch.load(vae_savepath))
                self.discriminator.load_state_dict(torch.load(disc_savepath))
                return
        return

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

    def summary(self):
        out = []

        out += ["", "VAE:"]
        out += [f"Parameter Count = {sum(np.prod(p.size()) for p in self.vae.parameters() if p.requires_grad)}"]
        out += [str(self.vae)]

        out += ["", "Discriminator:"]
        out += [f"Parameter Count = {sum(np.prod(p.size()) for p in self.discriminator.parameters() if p.requires_grad)}"]
        out += [str(self.discriminator)]

        return "\n".join(out)

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
        self.epoch_size += minibatch.shape[0]
        self.epoch_vae_loss += vae_loss.item()
        self.epoch_vae_recon_loss += recon_loss.item()
        self.epoch_vae_kl_loss += kl_loss.item()
        self.epoch_vae_gan_loss += gan_loss.item()
        self.epoch_disc_loss += discriminator_loss.item()

    def log_metrics(self, epoch_idx, val_data, epoch_complete=True):
        super(VaeGanHandler, self).log_metrics(epoch_idx, val_data, epoch_complete=epoch_complete)

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

