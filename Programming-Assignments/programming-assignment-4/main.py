# main.py

import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import gans

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Train GANs")
    parser.add_argument('--ngpu', default=0, help="number of GPUs for training")
    parser.add_argument('--gan_type', default='CycleGAN', choices=["VanillaGAN", "CycleGAN"], help="GAN type to run")
    parser.add_argument('--nepochs', default=200, type=int, help="Number of epochs to train")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--save_dir', default='checkpoints', help="directory to save results")
    parser.add_argument('--logs_dir', default='logs', help="directory to save results")
    parser.add_argument('--project_name', default='Image2Image', help="Name of the Project")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    parser.add_argument('--optimizer', default='Adam', help="Optimization method")

    # Model hyper-parameters
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')        
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')    
    parser.add_argument('--lambda_cycle', type=float, default=0.015, help='weight parameter for cycle loss')

    # Training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--spectral_norm', type=bool, default=False, help='on/off for spectral normalization')
    parser.add_argument('--gradient_penalty', type=bool, default=False, help='on/off for gradient_penalty')
    parser.add_argument('--d_lr_factor', type=int, default=1)

    # Data sources
    parser.add_argument('--data_x', type=str, default='Apple', choices=['Apple', 'Windows'], help='Choose the type of images for domain x.')
    parser.add_argument('--data_y', type=str, default='Windows', choices=['Apple', 'Windows'], help='Choose the type of images for domain y.')

    args = parser.parse_args()

    if (args.ngpu == 0):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    name = 'results/{}'.format(args.gan_type)
    logger = TensorBoardLogger(
        save_dir=os.path.join(name, args.logs_dir),
        log_graph=True,
        name=args.project_name
    )

    model = getattr(gans, args.gan_type)(args)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(name, args.save_dir),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.6f}'
        )

    if args.ngpu == 0:
        strategy = None
        sync_batchnorm = False
    elif args.ngpu > 1:
        strategy = 'ddp'
        sync_batchnorm = True
    else:
        strategy = 'dp'
        sync_batchnorm = False

    trainer = pl.Trainer(
        gpus=args.ngpu,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        callbacks=[checkpoint],
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision
    )

trainer.fit(model)