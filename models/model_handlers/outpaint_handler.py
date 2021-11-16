from typing import Tuple
from .model_handler_base import *

class OutPaintHandler(ModelHandlerBase):
    def __init__(self, name: str, img_dims: Tuple[int, int], region_dims: Tuple[float, float], generator, local_discriminator, global_discriminator, weight_decay=0):
        super(OutPaintHandler, self).__init__(name, img_dims, region_dims)

        # Networks
        self.gen = generator.to(DEVICE)
        self.l_disc = local_discriminator.to(DEVICE)
        self.g_disc = global_discriminator.to(DEVICE)

        # Optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), weight_decay=weight_decay)
        self.l_disc_opt = torch.optim.Adam(self.l_disc.parameters())
        self.g_disc_opt = torch.optim.Adam(self.g_disc.parameters())

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
        super(OutPaintHandler, self).train_minibatch(minibatch)

        minibatch = minibatch.reshape(minibatch.shape[:1] + (1,) + minibatch.shape[1:])
        y = torch.tensor(minibatch).to(DEVICE)
        x = torch.tensor_split(y, 2, dim=2)[0] # Each input is only left half of full image

        # Discriminator Update
        for _ in range(self.DISCRIMINATOR_TRAINING_MULTIPLIER):
            self.l_disc_opt.zero_grad()
            l_disc_loss = self.local_discriminator_loss_fn(x, y)
            l_disc_loss.backward()
            self.l_disc_opt.step()

            self.g_disc_opt.zero_grad()
            g_disc_loss = self.global_discriminator_loss_fn(x, y)
            g_disc_loss.backward()
            self.g_disc_opt.step()

        # Generator Update
        self.gen_opt.zero_grad()
        gen_recon_loss, gen_l_disc_loss, gen_g_disc_loss = self.generator_loss_fn(x, y)
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
            self.l_disc_opt.zero_grad()
            self.g_disc_opt.zero_grad()

            batch_size = min(VAL_BATCH_SIZE, val_data.shape[0] - batch_start)
            val_batch = val_data[batch_start : batch_start + batch_size]
            y = torch.tensor(val_batch.reshape(val_batch.shape[:1] + (1,) + val_batch.shape[1:])).to(DEVICE)
            x = torch.tensor_split(y, 2, dim=2)[0]

            l_disc_loss = self.local_discriminator_loss_fn(x, y, test=True)
            g_disc_loss = self.global_discriminator_loss_fn(x, y, test=True)
            gen_recon_loss, gen_l_disc_loss, gen_g_disc_loss = self.generator_loss_fn(x, y)
            gen_loss = gen_recon_loss + gen_l_disc_loss + gen_g_disc_loss

            total_val_gen_loss += gen_loss.item()
            total_val_gen_recon_loss += gen_recon_loss.item()
            total_val_gen_l_disc_loss += gen_l_disc_loss.item()
            total_val_gen_g_disc_loss += gen_g_disc_loss.item()
            total_val_l_disc_loss += l_disc_loss.item()
            total_val_g_disc_loss += g_disc_loss.item()

            batch_start += VAL_BATCH_SIZE

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
            net_input = torch.tensor_split(torch.tensor(example_imgs.reshape(example_imgs.shape[:1] + (1,) + example_imgs.shape[1:])).to(DEVICE), 2, dim=2)[0]
            net_output = self.gen(net_input)
            example_reconstructions = net_output.detach().cpu().numpy().reshape(example_imgs.shape)

        data.util.save_image_lists([example_imgs, example_reconstructions], ["Original", "Generated"],
                                    main_title="{self.name}_{epoch_name}", filename=os.path.join(self.log_folder(), f"{epoch_name}.png"))

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

    def local_discriminator_loss_fn(self, x, y, test=False):
        full_real = y
        right_real = torch.tensor_split(full_real, 2, dim=2)[1]
        full_fake = self.gen(x)
        right_fake = torch.tensor_split(full_fake, 2, dim=2)[1]

        # Hinge Loss
        if self.DISCRIMINATOR_LOSS == "hinge":
            pos_hinge_loss = torch.mean(torch.relu(1 - self.l_disc(right_real)), dim=1)
            neg_hinge_loss = torch.mean(torch.relu(1 + self.l_disc(right_fake)), dim=1)

            l_disc_loss = torch.sum(
                0.5 * pos_hinge_loss + 0.5 * neg_hinge_loss
            )

        # Unbounded Loss
        elif self.DISCRIMINATOR_LOSS == "unbounded":
            l_disc_loss = -torch.sum(torch.mean(self.l_disc(right_real) - self.l_disc(right_fake), dim=1))

        
        # Gradient penalty
        if self.DISCRIMINATOR_REGULARIZER == "grad":
            img_differences = right_fake - right_real
            alpha = torch.rand(x.shape[0], 1, 1, 1).to(DEVICE) # Random perturbations of real image towards fake image
            perturbed_inputs = right_real + alpha * img_differences
            disc_on_perturbed = torch.sum(torch.mean(self.l_disc(perturbed_inputs), dim=1)) # sum over batches will be undone when we take derivative over batch inputs
            gradients = torch.autograd.grad(outputs=disc_on_perturbed, inputs=[perturbed_inputs], create_graph=not test)[0]
            slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=(1, 2, 3)) + 1e-10)
            gradient_penalty = self.DISCRIMINATOR_GRADIENT_PENALTY_WEIGHT * torch.sum(torch.square(slopes - 1))

            l_disc_loss += gradient_penalty

        return l_disc_loss

    def global_discriminator_loss_fn(self, x, y, test=False):
        full_real = y
        full_fake = self.gen(x)

        # Hinge Loss
        if self.DISCRIMINATOR_LOSS == "hinge":
            pos_hinge_loss = torch.mean(torch.relu(1 - self.g_disc(full_real)), dim=1)
            neg_hinge_loss = torch.mean(torch.relu(1 + self.g_disc(full_fake)), dim=1)

            g_disc_loss = torch.sum(
                0.5 * pos_hinge_loss + 0.5 * neg_hinge_loss
            )

        # Unbounded Loss
        if self.DISCRIMINATOR_LOSS == "unbounded":
            g_disc_loss = -torch.sum(torch.mean(self.g_disc(full_real) - self.g_disc(full_fake), dim=1))

        # Gradient penalty
        if self.DISCRIMINATOR_REGULARIZER == "grad":
            img_differences = full_fake - full_real
            alpha = torch.rand(x.shape[0], 1, 1, 1).to(DEVICE) # Random perturbations of real image towards fake image
            perturbed_inputs = full_real + alpha * img_differences
            disc_on_perturbed = torch.sum(torch.mean(self.g_disc(perturbed_inputs), dim=1)) # sum over batches will be undone when we take derivative over batch inputs
            gradients = torch.autograd.grad(outputs=disc_on_perturbed, inputs=[perturbed_inputs], create_graph=not test)[0]
            slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=(1, 2, 3)) + 1e-10)
            gradient_penalty = self.DISCRIMINATOR_GRADIENT_PENALTY_WEIGHT * torch.sum(torch.square(slopes - 1))

            g_disc_loss += gradient_penalty

        return g_disc_loss

    def generator_loss_fn(self, x, y):
        generated = self.gen(x)
        generated_right = torch.tensor_split(generated, 2, dim=2)[1]

        recon_loss = self.GEN_RECON_RELATIVE_LOSS_WEIGHT * torch.sum(
            torch.mean(torch.abs(generated - y) * self.recon_loss_mask, dim=(1, 2, 3))
        )

        l_disc_loss = self.GEN_L_DISC_RELATIVE_LOSS_WEIGHT * torch.sum(
            -torch.mean(self.l_disc(generated_right), dim=1)
        )
        g_disc_loss = self.GEN_G_DISC_RELATIVE_LOSS_WEIGHT * torch.sum(
            -torch.mean(self.g_disc(generated), dim=1)
        )

        return recon_loss, l_disc_loss, g_disc_loss