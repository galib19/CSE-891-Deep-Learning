# vanilla.py

import os
import torch
import numpy as np
import pytorch_lightning as pl

import imageio
from models import *
from dataloader import EmojiData

__all__ = ['VanillaGAN']

class VanillaGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.train_dataloader = EmojiData(args).train_dataloader
        self.discriminator = DCDiscriminator(conv_dim=self.hparams.d_conv_dim)
        self.generator = DCGenerator(noise_size=self.hparams.noise_size, conv_dim=self.hparams.g_conv_dim, spectral_norm=self.hparams.spectral_norm)

        self.fixed_noise = (torch.rand(self.hparams.batch_size, self.hparams.noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3)
        self.gp_weight = 10

        self.path = os.path.join('results/{}'.format(self.hparams.gan_type), 'images')
        os.makedirs(self.path, exist_ok=False)

    def sample_noise(self, batch_size):
        return (torch.rand(batch_size, self.hparams.noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3).to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch[0]
        batch_size = images.size(0)

        # train generator
        if optimizer_idx == 0:
            # FILL THIS IN
            # 1. Sample noise
            noise = self.sample_noise(batch_size)

            # 2. Generate fake images from the noise
            fake_images = self.generator(noise)

            # 3. Compute the generator loss
            g_loss = sum((self.discriminator(fake_images) - 1)**2)/batch_size

            # log sampled images
            grid = self.create_image_grid(fake_images.data.cpu().numpy())
            grid = np.uint8(255 * (grid + 1) / 2)
            self.logger.experiment.add_image("generated_images", np.moveaxis(grid, -1, 0), self.current_epoch)

            self.log('g_loss', g_loss, on_step=False, on_epoch=True, logger=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            d_real_loss = sum((self.discriminator(images) - 1)**2)/(batch_size)

            # 2. Sample noise
            noise = self.sample_noise(batch_size)

            # 3. Generate fake images from the noise
            fake_images = self.generator(noise)

            # 4. Compute the discriminator loss on the fake images
            d_fake_loss = sum(self.discriminator(fake_images)**2)/(2*batch_size)

            # ---- Gradient Penalty ----
            if self.hparams.gradient_penalty:
                alpha = torch.rand(images.shape[0], 1, 1, 1).to(self.device)
                alpha = alpha.expand_as(images)
                interp_images = (alpha * images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                D_interp_output = self.discriminator(interp_images.to(self.device))

                gradients = torch.autograd.grad(outputs=D_interp_output, inputs=interp_images,
                                                grad_outputs=torch.ones(D_interp_output.size()).to(self.device),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.view(images.shape[0], -1).to(self.device)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

                gp = self.gp_weight * gradients_norm.mean()
            else:
                gp = 0.0

            # 5. Compute the total discriminator loss
            d_total_loss = d_real_loss + d_fake_loss + gp

            self.log('d_loss', d_total_loss, on_step=False, on_epoch=True, logger=True)
            return d_total_loss

    def on_epoch_end(self):
        generated_images = self.generator(self.fixed_noise.to(self.device))

        grid = self.create_image_grid(generated_images.data.cpu().numpy())
        grid = np.uint8(255 * (grid + 1) / 2)

        filename = os.path.join(self.path, 'sample-{:06d}.png'.format(self.current_epoch))
        imageio.imsave(filename, grid)

    def create_image_grid(self, array, ncols=None):
        """
        """
        num_images, channels, cell_h, cell_w = array.shape

        if not ncols:
            ncols = int(np.sqrt(num_images))
        nrows = int(np.math.floor(num_images / float(ncols)))
        result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
        for i in range(0, nrows):
            for j in range(0, ncols):
                result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2, 0)

        if channels == 1:
            result = result.squeeze()
        return result

    def configure_optimizers(self):
        g_optimizer = getattr(torch.optim, self.hparams.optimizer)(self.generator.parameters(), lr=self.hparams.lr)
        d_optimizer = getattr(torch.optim, self.hparams.optimizer)(self.discriminator.parameters(), lr=self.hparams.lr * self.hparams.d_lr_factor)
        return [g_optimizer, d_optimizer], []
