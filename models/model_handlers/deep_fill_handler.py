from .model_handler_base import *

class DeepFillHandler(ModelHandlerBase):
    def __init__(self, name, img_dims, region_dims, generator, discriminator):
        super(DeepFillHandler, self).__init__(name, img_dims, region_dims)

        # Networks
        self.gen = generator.to(DEVICE)
        self.disc = discriminator.to(DEVICE)

        # Optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters())
        self.disc_opt = torch.optim.Adam(self.disc.parameters())

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

        # Logging count - allows mid-epoch tensorboard logging
        self.log_step = 0

    def load_weights(self):
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
        out += [f"Parameter Count = {sum(np.prod(p.size()) for p in self.gen.parameters() if p.requires_grad)}"]
        out += [str(self.gen)]

        out += ["", "Discriminator:"]
        out += [f"Parameter Count = {sum(np.prod(p.size()) for p in self.disc.parameters() if p.requires_grad)}"]
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
        super(DeepFillHandler, self).log_metrics(epoch_idx, val_data, epoch_complete=epoch_complete)

        # Find val loss 
        total_val_gen_loss = 0
        total_val_gen_recon_loss = 0
        total_val_gen_gan_loss = 0
        total_val_gen_loss = 0
        total_val_disc_loss = 0

        batch_start = 0
        with torch.no_grad():
            while batch_start < val_data.shape[0]:
                batch_size = min(VAL_BATCH_SIZE, val_data.shape[0] - batch_start)
                val_batch = val_data[batch_start : batch_start + batch_size]
                x = torch.tensor(val_batch.reshape(val_batch.shape[:1] + (1,) + val_batch.shape[1:])).to(DEVICE)
                
                # Add masks
                mask = self.create_mask(batch_size)

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
        self.writer.add_scalar("eval/gen_loss", val_gen_loss, self.log_step)#, epoch_idx)
        self.writer.add_scalar("eval/gen_recon_loss", val_gen_recon_loss, self.log_step)#, epoch_idx)
        self.writer.add_scalar("eval/gen_gan_loss", val_gen_gan_loss, self.log_step)#, epoch_idx)
        self.writer.add_scalar("eval/disc_loss", val_disc_loss, self.log_step)#, epoch_idx)

        # Log train loss
        if self.epoch_size != 0:
            self.writer.add_scalar("train/gen_loss", self.epoch_gen_loss / self.epoch_size, self.log_step)#, epoch_idx)
            self.writer.add_scalar("train/gen_recon_loss", self.epoch_gen_recon_loss / self.epoch_size, self.log_step)#, epoch_idx)
            self.writer.add_scalar("train/gen_gan_loss", self.epoch_gen_gan_loss / self.epoch_size, self.log_step)#, epoch_idx)
            self.writer.add_scalar("train/disc_loss", self.epoch_disc_loss / self.epoch_size, self.log_step)#, epoch_idx)

        # Increment log step
        self.log_step += 1

        # Save some examples from the val_data
        epoch_name = f"epoch_{epoch_idx:.2f}"
        rand_inds = np.random.choice(np.arange(val_data.shape[0]), 20, replace=False)
        example_imgs = val_data[rand_inds]

        with torch.no_grad():
            example_inputs = torch.tensor(example_imgs.reshape(example_imgs.shape[:1] + (1,) + example_imgs.shape[1:])).to(DEVICE)
            example_masks = self.create_mask(example_imgs.shape[0])
            _, final_output = self.gen(example_inputs, example_masks)
            example_reconstructions = final_output.detach().cpu().numpy().reshape(example_imgs.shape)
            example_masked_imgs = ((1 - example_masks) * example_inputs + example_masks * 1.1 * torch.max(torch.abs(example_inputs))).detach().cpu().numpy().reshape(example_imgs.shape)

        titles = lambda col, row : ["Original", "Masked", "Reconstruction"][col]
        data.util.save_image_lists([example_imgs, example_masked_imgs, example_reconstructions], titles,
                                    main_title=f"{self.name}_{epoch_name}", filename=os.path.join(self.log_folder(), f"{epoch_name}.png"))

        # Reset train loss vars
        if epoch_complete:
            self.epoch_size = 0
            self.epoch_vae_loss = 0
            self.epoch_vae_recon_loss = 0
            self.epoch_vae_kl_loss = 0
            self.epoch_vae_gan_loss = 0
            self.epoch_disc_loss = 0






    # --- Private Helper Functions ---

    '''
    Original implementation uses GAN hinge loss
    '''
    def discriminator_loss_fn(self, img, mask):
        coarse_generated, refined_generated = self.gen(img, mask)

        # Only use generator output within masked space for discriminator
        patched_generated = refined_generated * mask + img * (1 - mask)

        pos_hinge_loss = torch.mean(torch.relu(1 - self.disc(img, mask)), dim=1)
        neg_hinge_loss = torch.mean(torch.relu(1 + self.disc(patched_generated, mask)), dim=1)


        disc_loss = torch.sum(
            0.5 * pos_hinge_loss + 0.5 * neg_hinge_loss
        )
        
        return disc_loss

    '''
    Original implementation uses:
        recon loss = l1 reconstruction loss on both coarse and refined generations
        gan loss = hinge loss from (https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py)
    '''
    def generator_loss_fn(self, img, mask):
        coarse_generated, refined_generated = self.gen(img, mask)

        # Only use generated output within masked space for discriminator
        patched_generated = refined_generated * mask + img * (1 - mask)

        recon_loss = self.GEN_RECON_RELATIVE_LOSS_WEIGHT * torch.sum(
            torch.mean(torch.abs(img - coarse_generated) + torch.abs(img - refined_generated), dim=(1, 2, 3))
        )
        gan_loss = self.GEN_GAN_RELATIVE_LOSS_WEIGHT * torch.sum(
            -torch.mean(self.disc(patched_generated, mask), dim=1)
        )
        
        return recon_loss, gan_loss

    def create_mask(self, batch_size):
        if self.MASK_TYPE == "fixed_inner_square":
            quarter_len_x, quarter_len_y = self.img_dims[0] // 4, self.img_dims[1] // 4

            mask = torch.zeros((batch_size, 1, *self.img_dims), device=DEVICE)
            mask[:, :, quarter_len_x : 3 * quarter_len_x, quarter_len_y : 3 * quarter_len_y] = torch.ones((batch_size, 1, 2 * quarter_len_x, 2 * quarter_len_y), device=DEVICE)
            return mask

        if self.MASK_TYPE == "shifting_inner_square":
            quarter_len_x, quarter_len_y = self.img_dims[0] // 4, self.img_dims[1] // 4

            x_mask_start = np.random.choice(range(2 * quarter_len_x), batch_size)
            y_mask_start = np.random.choice(range(2 * quarter_len_y), batch_size)

            mask = torch.zeros((batch_size, 1, *self.img_dims), device=DEVICE)
            for i in range(batch_size):
                mask[i, :, x_mask_start[i] : x_mask_start[i] + 2 * quarter_len_x, y_mask_start[i] : y_mask_start[i] + 2 * quarter_len_y] = torch.ones((1, 2 * quarter_len_x, 2 * quarter_len_y), device=DEVICE)
            return mask

        if self.MASK_TYPE == "corner_square":
            x_mask_start = np.random.choice([0, self.img_dims[0] // 2], batch_size)
            y_mask_start = np.random.choice([0, self.img_dims[1] // 2], batch_size)

            mask = torch.zeros((batch_size, 1, *self.img_dims), device=DEVICE)
            for i in range(batch_size):
                mask[i, :, x_mask_start[i] : x_mask_start[i] + self.img_dims[0] // 2, y_mask_start[i] : y_mask_start[i] + self.img_dims[1] // 2] = torch.ones((1, self.img_dims[0] // 2, self.img_dims[1] // 2), device=DEVICE)
            return mask

        if self.MASK_TYPE == "half_mask":
            base_mask = torch.zeros(1, *self.img_dims, device=DEVICE)
            base_mask[:, :self.img_dims[0] // 2, :] += torch.ones(1, 1, 1, device=DEVICE)
            possible_masks = [
                base_mask,
                base_mask.flip(1),
                base_mask.transpose(1, 2),
                base_mask.flip(1).transpose(1, 2)
            ]
            mask_choices = np.random.choice(range(4), batch_size)

            mask = torch.zeros((batch_size, 1, *self.img_dims), device=DEVICE)
            for i in range(batch_size):
                mask[i] = possible_masks[mask_choices[i]]
            
            return mask