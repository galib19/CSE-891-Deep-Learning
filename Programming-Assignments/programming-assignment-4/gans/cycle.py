# cycle.py

import os
import torch
import numpy as np
import pytorch_lightning as pl

import imageio
from models import *
from dataloader import EmojiData

__all__ =['CycleGAN']

class CycleGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.val_dataloader = EmojiData(args).val_dataloader
        self.train_dataloader = EmojiData(args).train_dataloader

        self.d_x = DCDiscriminator(conv_dim=self.hparams.d_conv_dim)
        self.d_y = DCDiscriminator(conv_dim=self.hparams.d_conv_dim)
        self.g_x2y = CycleGenerator(conv_dim=self.hparams.g_conv_dim, init_zero_weights=self.hparams.init_zero_weights, spectral_norm=self.hparams.spectral_norm)
        self.g_y2x = CycleGenerator(conv_dim=self.hparams.g_conv_dim, init_zero_weights=self.hparams.init_zero_weights, spectral_norm=self.hparams.spectral_norm)

        self.fixed_noise = (torch.rand(self.hparams.batch_size, self.hparams.noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3)
        self.path = os.path.join('results/{}'.format(self.hparams.gan_type), 'images')
        os.makedirs(self.path, exist_ok=False)

    def sample_noise(self):
        return (torch.rand(self.hparams.batch_size, self.hparams.noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3).to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images_x, _ = batch[0]
        images_y, _ = batch[1]
        batch_size = images_x.size(0)

        # train generator
        if optimizer_idx == 0:
            #########################################
            ##    FILL THIS IN: Y--X-->Y CYCLE     ##
            #########################################
            # 1. Generate fake images that look like domain X based on real images in domain Y
            fake_x = self.g_y2x(images_y)
            # 2. Compute the generator loss based on domain X
            g_loss = torch.sum(torch.pow(self.d_x(fake_x) - 1, 2))/(batch_size)

            reconstructed_Y = self.g_x2y(fake_x)

            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = (images_y - reconstructed_Y).abs().sum()/(images_y).size(0)
            g_loss += self.hparams.lambda_cycle * cycle_consistency_loss

            #########################################
            ##    FILL THIS IN: X--Y-->X CYCLE     ##
            #########################################

            # 1. Generate fake images that look like domain Y based on real images in domain X
            fake_y = self.g_x2y(images_x)

            # 2. Compute the generator loss based on domain Y
            g_loss = torch.sum(torch.pow(self.d_y(fake_y) - 1, 2))/(batch_size)

            reconstructed_x = self.g_y2x(fake_y)

            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = (images_x - reconstructed_x).abs().sum()/(images_x).size(0)
            g_loss += self.hparams.lambda_cycle * cycle_consistency_loss

            # log sampled images
            grid_xy, grid_yx = self.create_image_grids(images_y, images_x, fake_y, fake_x)
            self.logger.experiment.add_image("generated_images x->y", np.moveaxis(grid_xy, -1, 0), self.current_epoch)
            self.logger.experiment.add_image("generated_images y->x", np.moveaxis(grid_yx, -1, 0), self.current_epoch)

            self.log('g_loss', g_loss, on_step=False, on_epoch=True, logger=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # FILL THIS IN
            # 1. Compute the discriminator losses on real images
            d_x_loss = torch.sum(torch.pow(self.d_x(images_x) - 1, 2))/(2*batch_size)
            d_y_loss = torch.sum(torch.pow(self.d_y(images_y) - 1, 2))/(2*batch_size)
            d_real_loss = d_x_loss + d_y_loss

            # 2. Generate fake images that look like domain X based on real images in domain Y
            fake_x = self.g_y2x(images_y)

            # 3. Compute the loss for D_X
            d_x_loss = torch.sum(torch.pow(self.d_x(fake_x) - 1, 2))/(2*batch_size)

            # 4. Generate fake images that look like domain Y based on real images in domain X
            fake_y = self.g_x2y(images_x)

            # 5. Compute the loss for D_Y
            d_y_loss = torch.sum(torch.pow(self.d_y(fake_y) - 1, 2))/(2*batch_size)

            # 6. Compute the discriminator losses on fake images
            d_fake_loss = d_x_loss + d_y_loss

            # 7. Compute total discriminator loss
            d_total_loss = d_real_loss + d_fake_loss

            if batch_idx == 0:
                img1, img2 = self.create_image_grids(images_y, images_x, fake_y, fake_x)
                self.logger.experiment.add_image("xtoy", np.moveaxis(img1, -1, 0), self.current_epoch)
                self.logger.experiment.add_image("ytox", np.moveaxis(img2, -1, 0), self.current_epoch)


                filename = os.path.join(self.path, 'sample-{:06d}-X-Y.png'.format(self.current_epoch))
                imageio.imsave(filename, img1)
                filename = os.path.join(self.path, 'sample-{:06d}-Y-X.png'.format(self.current_epoch))
                imageio.imsave(filename, img2)

            self.log('d_loss', d_total_loss, on_step=False, on_epoch=True, logger=True)
            return d_total_loss

    def merge_images(self, sources, targets):
        """Creates a grid consisting of pairs of columns, where the first column in
        each pair contains images source images and the second column in each pair
        contains images generated by the CycleGAN from the corresponding images in
        the first column.
        """
        batch_size, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)


    def create_image_grids(self, y, x, fake_y, fake_x):
        """Saves samples from both generators X->Y and Y->X.
        """
        x = x.data.cpu().numpy()
        y = y.data.cpu().numpy()
        fake_x = fake_x.data.cpu().numpy()
        fake_y = fake_y.data.cpu().numpy()

        merged_xy = self.merge_images(x, fake_y)
        merged_xy = np.uint8(255 * ((merged_xy + 1)/2))

        merged_yx = self.merge_images(y, fake_x)
        merged_yx = np.uint8(255 * ((merged_yx + 1)/2))
        return merged_xy, merged_yx

    def configure_optimizers(self):
        g_params = list(self.g_x2y.parameters()) + list(self.g_y2x.parameters())  # Get generator parameters
        d_params = list(self.d_x.parameters()) + list(self.d_y.parameters())  # Get discriminator parameters
        g_optimizer = getattr(torch.optim, self.hparams.optimizer)(g_params, lr=self.hparams.lr)
        d_optimizer = getattr(torch.optim, self.hparams.optimizer)(d_params, lr=self.hparams.lr * self.hparams.d_lr_factor)
        return [g_optimizer, d_optimizer], []
