# colorization.py

import os
import argparse

import torch
import models
import numpy as np
import torch.nn as nn
from dataloaders import CIFARColorizationData
from torchvision.utils import make_grid, save_image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class CIFARColorization(pl.LightningModule):
    def __init__(self, args, dataloader, colors):
        super().__init__()
        self.args = args
        self.colors = colors
        self.num_colors = colors.shape[0]

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.test_dataloader = dataloader.test_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.model = getattr(models, args.model)(args.kernel, args.num_filters, self.num_colors, 1)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inp, gnd = batch
        batch_size = inp.size(0)
        out = self.model(inp)
        loss_out = out.transpose(1, 3).contiguous().view([batch_size * 32 * 32, self.num_colors])
        loss_lab = gnd.transpose(1, 3).contiguous().view([batch_size * 32 * 32])
        loss = self.criterion(loss_out, loss_lab)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, gnd = batch
        batch_size = inp.size(0)
        out = self.model(inp)
        loss_out = out.transpose(1, 3).contiguous().view([batch_size * 32 * 32, self.num_colors])
        loss_lab = gnd.transpose(1, 3).contiguous().view([batch_size * 32 * 32])
        loss = self.criterion(loss_out, loss_lab)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)

        if batch_idx == 0:
            _, out = torch.max(out.data, 1, keepdim=True)
            inp1 = make_grid(inp[:8].cpu())
            out1 = make_grid(self.colors[out[:8, 0]].transpose(3,1).cpu())
            gnd1 = make_grid(self.colors[gnd[:8, 0]].transpose(3,1).cpu())
            self.logger.experiment.add_image('input', inp1, self.current_epoch)
            self.logger.experiment.add_image('output', out1, self.current_epoch)
            self.logger.experiment.add_image('ground truth', gnd1, self.current_epoch)
            save_image(inp1, os.path.join(self.args.save_dir, self.args.model + "-" + 'colorization_inp.png'))
            save_image(out1, os.path.join(self.args.save_dir, self.args.model + "-" + 'colorization_out.png'))
            save_image(gnd1, os.path.join(self.args.save_dir, self.args.model + "-" + 'colorization_gnd.png'))

        return loss

    def testing_step(self, batch, batch_idx):
        inp, gnd = batch
        out = self.model(inp)
        loss = self.criterion(out, gnd)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.args.optimizer)(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learn_rate)
        return [optimizer]

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Train colorization")
    parser.add_argument('--ngpu', default=0, help="number of GPUs for training")
    parser.add_argument('--colors', default='colors/color_kmeans24_cat7.npy', help="Discrete color clusters to use")
    parser.add_argument('--model', choices=["PoolUpsampleNet", "ConvTransposeNet", "UNet"], help="Model to run")
    parser.add_argument('--kernel', default=3, type=int, help="Convolution kernel size")
    parser.add_argument('--num_filters', default=32, type=int, help="Base number of convolution filters")
    parser.add_argument('--learn_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch size")
    parser.add_argument('--nepochs', default=25, type=int, help="Number of epochs to train")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--save_dir', default='results/', help="directory to save results")
    parser.add_argument('--logs_dir', default='logs/', help="directory to save results")
    parser.add_argument('--data_dir', default='data/', help="directory to save downloaded data")
    parser.add_argument('--project_name', default='Colorization', help="Name of the Project")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    parser.add_argument('--optimizer', default='AdamW', help="Optimization method")

    args = parser.parse_args()

    if (args.ngpu == 0):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    logger = TensorBoardLogger(
        save_dir=args.logs_dir,
        log_graph=True,
        name=args.project_name
    )

    colors = torch.from_numpy(np.load(args.colors, allow_pickle=True, encoding="bytes")[0]).float()

    dataloader = CIFARColorizationData(args)
    dataloader.setup(split='train')
    model = CIFARColorization(args, dataloader, colors)

    checkpoint = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=5)

    if args.ngpu == 0:
        accelerator = None
        sync_batchnorm = False
    elif args.ngpu > 1:
        accelerator = 'ddp'
        sync_batchnorm = True
    else:
        accelerator = 'dp'
        sync_batchnorm = False

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator=accelerator,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        callbacks=[checkpoint],
        checkpoint_callback=True,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision
        )

    trainer.fit(model)
