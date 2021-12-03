from typing import Tuple
from .model_handler_base import *

'''
Handles training and testing for outpainting models, which generate an attached right-half image given a left-half image
'''

class OutPaintHandler(ModelHandlerBase):
    def __init__(self, name: str, img_dims: Tuple[int, int], region_dims: Tuple[float, float], generator, local_discriminator, global_discriminator, weight_decay=0):
        super(OutPaintHandler, self).__init__(name, img_dims, region_dims)

        # Networks
        self.gen = generator.to(DEVICE)
        self.l_disc = local_discriminator.to(DEVICE)
        self.g_disc = global_discriminator.to(DEVICE)

        # Optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), weight_decay=weight_decay)
        self.disc_opt = torch.optim.Adam(
            list(self.l_disc.parameters()) +
            list(self.g_disc.parameters())
        )

        # Loss Options
        self.DISCRIMINATOR_LOSS = "hinge" # options are: hinge, unbounded
        self.DISCRIMINATOR_REGULARIZER = "none" # options are: none, grad

        # Training Constants
        self.DISCRIMINATOR_TRAINING_MULTIPLIER = 2
        self.DISCRIMINATOR_GRADIENT_PENALTY_WEIGHT = 10

        self.GEN_RECON_RELATIVE_LOSS_WEIGHT = 1 # in NS-outpaint, disc loss to recon loss is 0.002:0.998, and within disc, global to local is 0.9:0.1
        self.GEN_L_DISC_RELATIVE_LOSS_WEIGHT = 0.0018
        self.GEN_G_DISC_RELATIVE_LOSS_WEIGHT = 0.0002

        # State variables - updated over each epoch
        self.epoch_size = 0
        self.epoch_gen_loss = 0
        self.epoch_gen_recon_loss = 0
        self.epoch_gen_l_disc_loss = 0
        self.epoch_gen_g_disc_loss = 0
        self.epoch_l_disc_loss = 0
        self.epoch_g_disc_loss = 0

        # Logging count - allows mid-epoch tensorboard logging
        self.log_step = 0

        # Save mask for weighing reconstruction loss
        # paper uses cosine curve
        # I added that the final column cannot be fully masked to zero (endpoint=True),
        # since that causes early generated outputs to have extreme values in the last generated image column
        mask_left = np.ones(img_dims[0] // 2)
        mask_right = mask_left * np.cos(np.linspace(0, np.pi / 2, img_dims[0] // 2, endpoint=False))
        mask = np.concatenate([mask_left, mask_right]).reshape(1, 1, -1, 1)
        self.recon_loss_mask = torch.from_numpy(mask).to(DEVICE)

        # Create function which converts a full image into the format expected for
        # the local discriminator (in this case, just returns the right (generated) half of the image)
        self.local_discriminator_formatter = lambda full_image : torch.tensor_split(full_image, 2, dim=2)[1]


    def load_weights(self):
        for prev_it in range(1, self.iteration)[::-1]:
            gen_savepath = os.path.join(self.log_folder(prev_it), "gen_weights.pth")
            l_disc_savepath = os.path.join(self.log_folder(prev_it), "l_disc_weights.pth")
            g_disc_savepath = os.path.join(self.log_folder(prev_it), "g_disc_weights.pth")
            if os.path.exists(gen_savepath) and os.path.exists(l_disc_savepath) and os.path.exists(g_disc_savepath):
                print(f"Loading weights from {self.name}, run {prev_it}")
                self.gen.load_state_dict(torch.load(gen_savepath))
                self.l_disc.load_state_dict(torch.load(l_disc_savepath))
                self.g_disc.load_state_dict(torch.load(g_disc_savepath))
                return
        return

    def save_weights(self):
        gen_savepath = os.path.join(self.log_folder(), "gen_weights.pth")
        l_disc_savepath = os.path.join(self.log_folder(), "l_disc_weights.pth")
        g_disc_savepath = os.path.join(self.log_folder(), "g_disc_weights.pth")
        torch.save(self.gen.state_dict(), gen_savepath)
        torch.save(self.l_disc.state_dict(), l_disc_savepath)
        torch.save(self.g_disc.state_dict(), g_disc_savepath)

    def status(self):
        message = "Gen Loss = {:.3f} ({:.3f}, {:.3f}, {:.3f}), Local-Disc Loss = {:.3f}, Global-Disc Loss = {:.3f}".format(
            self.epoch_gen_loss / self.epoch_size,
            self.epoch_gen_recon_loss / self.epoch_size,
            self.epoch_gen_l_disc_loss / self.epoch_size,
            self.epoch_gen_g_disc_loss / self.epoch_size,
            self.epoch_l_disc_loss / self.epoch_size,
            self.epoch_g_disc_loss / self.epoch_size,
        )
        return message

    def summary(self):
        out = []

        out += ["", "Generator:"]
        out += [f"Parameter Count = {sum(np.prod(p.shape) for p in self.gen.parameters() if p.requires_grad)}"]
        out += [str(self.gen)]

        out += ["", "Local Discriminator:"]
        out += [f"Parameter Count = {sum(np.prod(p.shape) for p in self.l_disc.parameters() if p.requires_grad)}"]
        out += [str(self.l_disc)]

        out += ["", "Global Discriminator:"]
        out += [f"Parameter Count = {sum(np.prod(p.shape) for p in self.g_disc.parameters() if p.requires_grad)}"]
        out += [str(self.g_disc)]

        return "\n".join(out)

    def train_minibatch(self, minibatch):
        minibatch = minibatch.reshape(minibatch.shape[:1] + (1,) + minibatch.shape[1:])
        full_images = torch.tensor(minibatch).to(DEVICE)

        # Discriminator Update
        for _ in range(self.DISCRIMINATOR_TRAINING_MULTIPLIER):
            self.disc_opt.zero_grad()
            l_disc_loss, g_disc_loss = self.discriminator_loss_fn_all(full_images)
            disc_loss = l_disc_loss + g_disc_loss
            disc_loss.backward()
            self.disc_opt.step()

        # Generator Update
        self.gen_opt.zero_grad()
        gen_recon_loss, gen_l_disc_loss, gen_g_disc_loss = self.generator_loss_fn(full_images)
        gen_loss = gen_recon_loss + gen_l_disc_loss + gen_g_disc_loss
        gen_loss.backward()
        self.gen_opt.step()

        # Save metrics
        self.epoch_size += minibatch.shape[0]
        self.epoch_gen_loss += gen_loss.item()
        self.epoch_gen_recon_loss += gen_recon_loss.item()
        self.epoch_gen_l_disc_loss += gen_l_disc_loss.item()
        self.epoch_gen_g_disc_loss += gen_g_disc_loss.item()
        self.epoch_l_disc_loss += l_disc_loss.item()
        self.epoch_g_disc_loss += g_disc_loss.item()

    def log_metrics(self, epoch_idx, val_data, epoch_complete=True):
        super(OutPaintHandler, self).log_metrics(epoch_idx, val_data, epoch_complete=epoch_complete)

        # Find val loss
        total_val_gen_loss = 0
        total_val_gen_recon_loss = 0
        total_val_gen_l_disc_loss = 0
        total_val_gen_g_disc_loss = 0
        total_val_l_disc_loss = 0
        total_val_g_disc_loss = 0

        batch_start = 0
        while batch_start < val_data.shape[0]:
            # Reset grads
            self.gen_opt.zero_grad()
            self.disc_opt.zero_grad()

            batch_size = min(VAL_BATCH_SIZE, val_data.shape[0] - batch_start)
            val_batch = val_data[batch_start : batch_start + batch_size]
            full_images = torch.tensor(val_batch.reshape(val_batch.shape[:1] + (1,) + val_batch.shape[1:])).to(DEVICE)

            l_disc_loss, g_disc_loss = self.discriminator_loss_fn_all(full_images, test=True)
            gen_recon_loss, gen_l_disc_loss, gen_g_disc_loss = self.generator_loss_fn(full_images)
            gen_loss = gen_recon_loss + gen_l_disc_loss + gen_g_disc_loss

            total_val_gen_loss += gen_loss.item()
            total_val_gen_recon_loss += gen_recon_loss.item()
            total_val_gen_l_disc_loss += gen_l_disc_loss.item()
            total_val_gen_g_disc_loss += gen_g_disc_loss.item()
            total_val_l_disc_loss += l_disc_loss.item()
            total_val_g_disc_loss += g_disc_loss.item()

            # Delete tensors to free space before next iteration
            del l_disc_loss, g_disc_loss, gen_loss, gen_recon_loss, gen_l_disc_loss, gen_g_disc_loss

            batch_start += self.batch_size

        val_gen_loss = total_val_gen_loss / val_data.shape[0]
        val_gen_recon_loss = total_val_gen_recon_loss / val_data.shape[0]
        val_gen_l_disc_loss = total_val_gen_l_disc_loss / val_data.shape[0]
        val_gen_g_disc_loss = total_val_gen_g_disc_loss / val_data.shape[0]
        val_l_disc_loss = total_val_l_disc_loss / val_data.shape[0]
        val_g_disc_loss = total_val_g_disc_loss / val_data.shape[0]

        # Log val loss
        self.writer.add_scalar("eval/gen_loss", val_gen_loss, self.log_step)#epoch_idx)
        self.writer.add_scalar("eval/gen_recon_loss", val_gen_recon_loss, self.log_step)#epoch_idx)
        self.writer.add_scalar("eval/gen_l_disc_loss", val_gen_l_disc_loss, self.log_step)#epoch_idx)
        self.writer.add_scalar("eval/gen_g_disc_loss", val_gen_g_disc_loss, self.log_step)#epoch_idx)
        self.writer.add_scalar("eval/l_disc_loss", val_l_disc_loss, self.log_step)#epoch_idx)
        self.writer.add_scalar("eval/g_disc_loss", val_g_disc_loss, self.log_step)#epoch_idx)

        # Log train loss
        if self.epoch_size != 0:
            self.writer.add_scalar("train/gen_loss", self.epoch_gen_loss / self.epoch_size, self.log_step)#epoch_idx)
            self.writer.add_scalar("train/gen_recon_loss", self.epoch_gen_recon_loss / self.epoch_size, self.log_step)#epoch_idx)
            self.writer.add_scalar("train/gen_l_disc_loss", self.epoch_gen_l_disc_loss / self.epoch_size, self.log_step)#epoch_idx)
            self.writer.add_scalar("train/gen_g_disc_loss", self.epoch_gen_g_disc_loss / self.epoch_size, self.log_step)#epoch_idx)
            self.writer.add_scalar("train/l_disc_loss", self.epoch_l_disc_loss / self.epoch_size, self.log_step)#epoch_idx)
            self.writer.add_scalar("train/g_disc_loss", self.epoch_g_disc_loss / self.epoch_size, self.log_step)#epoch_idx)

        # Increment log step
        self.log_step += 1

        # Save some examples from the val_data
        epoch_name = f"epoch_{epoch_idx:.2f}"
        rand_inds = np.random.choice(np.arange(val_data.shape[0]), 20)
        example_imgs = val_data[rand_inds]
        
        with torch.no_grad():
            net_input = torch.tensor(example_imgs.reshape(example_imgs.shape[:1] + (1,) + example_imgs.shape[1:])).to(DEVICE)
            net_output = self.gen(net_input)
            example_reconstructions = net_output.detach().cpu().numpy().reshape(example_imgs.shape)

        titles = lambda col, row : ["Original", "Generated"][col]
        data.util.save_image_lists([example_imgs, example_reconstructions], titles,
                                    main_title=f"{self.name}_{epoch_name}", filename=os.path.join(self.log_folder(), f"{epoch_name}.png"))

        # Reset train loss vars
        if epoch_complete:
            self.epoch_size = 0
            self.epoch_gen_loss = 0
            self.epoch_gen_recon_loss = 0
            self.epoch_gen_l_disc_loss = 0
            self.epoch_gen_g_disc_loss = 0
            self.epoch_l_disc_loss = 0
            self.epoch_g_disc_loss = 0




    # --- Private Helper Functions ---

    # Discriminator loss function which can be applied to any discriminator, after arbitrary data processing
    def generic_discriminator_loss_fn(self, real: torch.Tensor, fake: torch.Tensor, target_disc: nn.Module, test: bool = False) -> torch.Tensor:
        # Hinge Loss
        if self.DISCRIMINATOR_LOSS == "hinge":
            pos_hinge_loss = torch.mean(torch.relu(1 - target_disc(real)), dim=1)
            neg_hinge_loss = torch.mean(torch.relu(1 + target_disc(fake)), dim=1)

            disc_loss = torch.sum(
                0.5 * pos_hinge_loss + 0.5 * neg_hinge_loss
            )

        # Unbounded Loss
        elif self.DISCRIMINATOR_LOSS == "unbounded":
            disc_loss = -torch.sum(torch.mean(target_disc(real) - target_disc(fake), dim=1))

        
        # Gradient penalty
        if self.DISCRIMINATOR_REGULARIZER == "grad":
            alpha = torch.rand(fake.shape[0], 1, 1, 1).to(DEVICE) # Random perturbations of real image towards fake image
            perturbed_inputs = real + alpha * (fake - real)
            disc_on_perturbed = torch.sum(torch.mean(target_disc(perturbed_inputs), dim=1)) # sum over batches will be undone when we take derivative over batch inputs
            gradients = torch.autograd.grad(outputs=disc_on_perturbed, inputs=[perturbed_inputs], create_graph=not test)[0]
            slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=(1, 2, 3)) + 1e-10)
            gradient_penalty = self.DISCRIMINATOR_GRADIENT_PENALTY_WEIGHT * torch.sum(torch.square(slopes - 1))

            disc_loss += gradient_penalty

        return disc_loss

    # Computes the loss function for both local and global discriminator
    def discriminator_loss_fn_all(self, full_image: torch.Tensor, test: bool = False) -> torch.Tensor:
        global_real = full_image
        global_fake = self.gen(full_image)
        local_real = self.local_discriminator_formatter(global_real)
        local_fake = self.local_discriminator_formatter(global_fake)

        l_disc_loss = self.generic_discriminator_loss_fn(local_real, local_fake, self.l_disc, test=test)
        g_disc_loss = self.generic_discriminator_loss_fn(global_real, global_fake, self.g_disc, test=test)

        return l_disc_loss, g_disc_loss

    def generator_loss_fn(self, full_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generated = self.gen(full_image)

        recon_loss = self.GEN_RECON_RELATIVE_LOSS_WEIGHT * torch.sum(
            torch.mean(torch.abs(generated - full_image) * self.recon_loss_mask, dim=(1, 2, 3))
        )

        local_generated = self.local_discriminator_formatter(generated)
        l_disc_loss = self.GEN_L_DISC_RELATIVE_LOSS_WEIGHT * torch.sum(
            -torch.mean(self.l_disc(local_generated), dim=1)
        )

        global_generated = generated
        g_disc_loss = self.GEN_G_DISC_RELATIVE_LOSS_WEIGHT * torch.sum(
            -torch.mean(self.g_disc(global_generated), dim=1)
        )

        return recon_loss, l_disc_loss, g_disc_loss