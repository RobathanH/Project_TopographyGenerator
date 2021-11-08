from .model_handler_base import *

class DeepFillHandler(ModelHandlerBase):
    def __init__(self, name, img_dims, region_dims, generator, discriminator):
        super(DeepFillHandler, self).__init__(name, img_dims, region_dims)

        # Networks
        self.gen = generator
        self.disc = discriminator

        # Optimizers
        self.gen = torch.optim.Adam(self.gen.parameters())
        self.disc = torch.optim.Adam(self.disc.parameters())

        # Training Constants
        self.DISCRIMINATOR_TRAINING_MULTIPLIER = 1
        self.GEN_RECON_RELATIVE_LOSS_WEIGHT = 1
        self.GEN_GAN_RELATIVE_LOSS_WEIGHT = 1
        self.MASK_TYPE = "fixed_inner_square"
        
        # State variables - updated over each epoch
        self.epoch_size = 0
        self.epoch_gen_loss = 0
        self.epoch_gen_recon_loss = 0
        self.epoch_gen_gan_loss = 0
        self.epoch_disc_loss = 0

    def load_Weights(self):
        for prev_it in range(1, self.iteration)[::-1]:
            gen_savepath = os.path.join(self.log_folder(prev_it), "gen_weights.pth")
            disc_savepath = os.path.join(self.log_folder(prev_it), "disc_weights.pth")
            if os.path.exists(gen_savepath) and os.path.exists(disc_savepath):
                print(f"Loading weights from {self.name}, run {prev_it}")
                self.gen.load_state_dict(torch.load(gen_savepath))
                self.disc.load_state_dict(torch.load(disc_savepath))
                return
        return

    def save_weights(self):
        gen_savepath = os.path.join(self.log_folder(), "gen_weights.pth")
        disc_savepath = os.path.join(self.log_folder(), "disc_weights.pth")
        torch.save(self.gen.state_dict(), gen_savepath)
        torch.save(self.disc.state_dict(), disc_savepath)

    def status(self):
        message = "Gen Loss = {:.3f} ({:.3f}, {:.3f}), Disc Loss = {:.3f}".format(
            self.epoch_gen_loss / self.epoch_size,
            self.epoch_gen_recon_loss / self.epoch_size,
            self.epoch_gen_gan_loss / self.epoch_size,
            self.epoch_disc_loss / self.epoch_size
        )
        return message

    def summary(self):
        out = []

        out += ["", "Generator:"]
        out += [f"Parameter Count = {sum(p for p in self.gen.parameters() if p.requires_grad)}"]
        out += [str(self.gen)]

        out += ["", "Discriminator:"]
        out += [f"Parameter Count = {sum(p for p in self.gen.parameters() if p.requires_grad)}"]
        out += [str(self.disc)]

        return "\n".join(out)

    def train_minibatch(self, minibatch):
        minibatch = minibatch.reshape(minibatch.shape[:1] + (1,) + minibatch.shape[1:])
        x = torch.tensor(minibatch).to(DEVICE)

        # Add masks
        mask = self.create_mask(x.shape[0])

        # Discriminator Update
        for _ in range(self.DISCRIMINATOR_TRAINING_MULTIPLIER):
            self.disc_opt.zero_grad()
            disc_loss = self.discriminator_loss_fn(x, mask)
            disc_loss.backward()
            self.disc_opt.step()

        # Generator Update
        self.gen_opt.zero_grad()
        gen_recon_loss, gen_gan_loss = self.generator_loss_fn(x, mask)
        gen_loss = gen_recon_loss + gen_gan_loss
        gen_loss.backward()
        self.gen_opt.step()

        # Save metrics
        self.epoch_size += minibatch.shape[0]
        self.epoch_gen_loss += gen_loss.item()
        self.epoch_gen_recon_loss += gen_recon_loss.item()
        self.epoch_gen_gan_loss += gen_gan_loss.item()
        self.epoch_disc_loss += disc_loss.item()

    def log_metrics(self, epoch_idx, val_data, epoch_complete=True):
        # Find val loss 
        total_val_gen_loss = 0
        total_val_gen_recon_loss = 0
        total_val_gen_gan_loss = 0
        total_val_gen_loss = 0
        total_val_disc_loss = 0

        batch_start = 0
        while batch_start < val_data.shape[0]:
            batch_size = min(VAL_BATCH_SIZE, val_data.shape[0] - batch_start)
            val_batch = val_data[batch_start : batch_start + batch_size]
            x = torch.tensor(val_batch.reshape(val_batch.shape[:1] + (1,) + val_batch[1:])).to(DEVICE)
            
            # Add masks
            mask = self.mask(batch_size)

            disc_loss = self.discriminator_loss_fn(x, mask)
            gen_recon_loss, gen_gan_loss = self.generator_loss_fn(x, mask)
            gen_loss = gen_recon_loss + gen_gan_loss

            total_val_gen_loss += gen_loss.item()
            total_val_gen_recon_loss += gen_recon_loss.item()
            total_val_gen_gan_loss += gen_gan_loss.item()
            total_val_disc_loss += disc_loss.item()

            batch_start += VAL_BATCH_SIZE

        val_gen_loss = total_val_gen_loss / val_data.shape[0]
        val_gen_recon_loss = total_val_gen_recon_loss / val_data.shape[0]
        val_gen_gan_loss = total_val_gen_gan_loss / val_data.shape[0]
        val_disc_loss = total_val_disc_loss / val_data.shape[0]

        # Log val loss
        self.writer.add_scalar("eval/gen_loss", val_gen_loss, epoch_idx)
        self.writer.add_scalar("eval/gen_recon_loss", val_gen_recon_loss, epoch_idx)
        self.writer.add_scalar("eval/gen_gan_loss", val_gen_gan_loss, epoch_idx)
        self.writer.add_scalar("eval/disc_loss", val_disc_loss, epoch_idx)

        # Log train loss
        self.writer.add_scalar("train/gen_loss", self.epoch_gen_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/gen_recon_loss", self.epoch_gen_recon_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/gen_gan_loss", self.epoch_gen_gan_loss / self.epoch_size, epoch_idx)
        self.writer.add_scalar("train/disc_loss", self.epoch_disc_loss / self.epoch_size, epoch_idx)

        # Save some examples from the val_data
        epoch_name = f"epoch{round(epoch_idx, 2)}"
        rand_inds = np.random.choice(np.arange(val_data.shape[0]), 20)
        example_imgs = val_data[rand_inds]
        net_input = torch.tensor(example_imgs.reshape(example_imgs.shape[:1] + (1,) + example_imgs.shape[1:])).to(DEVICE)
        net_output, _, _ = self.vae(net_input)
        example_reconstructions = net_output.detach().cpu().numpy().reshape(example_imgs.shape)
        data.util.save_images2(example_imgs, example_reconstructions, main_title=f"{self.name}_{epoch_name}",
                                title1="Original", title2="Reconstruction", filename=os.path.join(self.log_folder(), f"{epoch_name}.png"))

        # Reset train loss vars
        if epoch_complete:
            self.epoch_size = 0
            self.epoch_vae_loss = 0
            self.epoch_vae_recon_loss = 0
            self.epoch_vae_kl_loss = 0
            self.epoch_vae_gan_loss = 0
            self.epoch_disc_loss = 0






    # --- Private Helper Functions ---
    def discriminator_loss_fn(self, img, mask):
        generated = self.gen(img, mask)

        disc_loss = (torch.sum(-torch.log(self.disc(img, mask))) + torch.sum(-torch.log(1 - self.disc(generated, mask)))) / 2
        
        return disc_loss

    def generator_loss_fn(self, img, mask):
        generated = self.gen(img, mask)

        recon_loss = self.GEN_RECON_RELATIVE_LOSS_WEIGHT * torch.sum(torch.mean((img - generated).pow(2) / 2), axis=(1, 2, 3))
        gan_loss = self.GEN_GAN_RELATIVE_LOSS_WEIGHT * -torch.sum(torch.log(self.disc(img, mask)))
        
        return recon_loss, gan_loss

    def create_mask(self, batch_size):
        if self.MASK_TYPE == "fixed_inner_square":
            quarter_len_x, quarter_len_y = self.img_dims[0] // 4, self.img_dims[1] // 4

            mask = torch.zeros((batch_size, 1, *self.img_dims))
            mask[:, :, quarter_len_x : 3 * quarter_len_x, quarter_len_y : 3 * quarter_len_y] = torch.ones((batch_size, 1, 2 * quarter_len_x, 2 * quarter_len_y))
            mask = mask.to(DEVICE)
            return mask