# segmentation.py

import os
import argparse

import torch
import torch.nn as nn
from dataloaders import OxfordFlowersData

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import losses
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class Segmentation(pl.LightningModule):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.test_dataloader = dataloader.test_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.model = getattr(models, args.model)()
        if args.loss == 'cross-entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss == 'soft-iou':
            self.criterion = losses.SoftIoU()
        self.metric = torchmetrics.IoU(2)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        self.colors = [i for color in colors for i in color]
        self.trans = transforms.ToTensor()

    def training_step(self, batch, batch_idx):
        inp, gnd = batch
        out = self.model(inp)
        loss = self.criterion(out['out'], gnd)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, gnd = batch
        out = self.model(inp)
        loss = self.criterion(out['out'], gnd)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)

        if batch_idx == 0:
            out = out['out'].argmax(1).unsqueeze(1)
            gnd = gnd.unsqueeze(1)
            inp1 = make_grid(inp[:4].cpu())
            out1 = make_grid(out[:4].cpu()).float()
            gnd1 = make_grid(gnd[:4].cpu()).float()
            self.logger.experiment.add_image('input', inp1, self.current_epoch)
            self.logger.experiment.add_image('output', out1, self.current_epoch)
            self.logger.experiment.add_image('ground truth', gnd1, self.current_epoch)
            save_image(inp1, os.path.join(self.args.save_dir, self.args.loss + "-" + 'segmentation_inp.png'))
            save_image(out1, os.path.join(self.args.save_dir, self.args.loss + "-" + 'segmentation_out.png'))
            save_image(gnd1, os.path.join(self.args.save_dir, self.args.loss + "-" + 'segmentation_gnd.png'))

        return loss

    def testing_step(self, batch, batch_idx):
        inp, gnd = batch
        out = self.model(inp)
        loss = self.criterion(out['out'], gnd)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.args.optimizer)(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learn_rate)
        return [optimizer]

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Fine Tune Segmentation")
    parser.add_argument('--ngpu', default=0, help="number of GPUs for training")
    parser.add_argument('--model', default="DeepLabV3", help="loss to optimize")
    parser.add_argument('--loss', choices=["cross-entropy", "soft-iou"], help="loss to optimize")
    parser.add_argument('--learn_rate', default=0.05, type=float, help="Learning rate")
    parser.add_argument('--batch_size_train', default=128, type=int, help="training batch size")
    parser.add_argument('--batch_size_val', default=256, type=int, help="validation batch size")
    parser.add_argument('--nepochs', default=10, type=int, help="Number of epochs to fine-tune")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--save_dir', default='results/', help="directory to save results")
    parser.add_argument('--logs_dir', default='logs/', help="directory to save results")
    parser.add_argument('--data_dir', default='data/', help="directory to save downloaded data")
    parser.add_argument('--project_name', default='Segmentation', help="Name of the Project")
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

    dataloader = OxfordFlowersData(args)
    dataloader.setup(split='train')
    model = Segmentation(args, dataloader)

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
