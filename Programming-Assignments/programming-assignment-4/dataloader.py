# dataloader.py

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

__all__ = ['EmojiData']


class EmojiData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """Creates training and test data loaders.
        """
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_dataset = []
        self.train_dataset = []

        if args.gan_type == 'VanillaGAN':
            train_path = os.path.join('./emojis', args.data_x)
            test_path = os.path.join('./emojis', 'Test_{}'.format(args.data_x))
            self.test_dataset.append(datasets.ImageFolder(test_path, transform))
            self.train_dataset.append(datasets.ImageFolder(train_path, transform))

        if args.gan_type == 'CycleGAN':
            train_path = os.path.join('./emojis', args.data_x)
            test_path = os.path.join('./emojis', 'Test_{}'.format(args.data_x))
            self.train_dataset.append(datasets.ImageFolder(train_path, transform))
            self.test_dataset.append(datasets.ImageFolder(test_path, transform))
            train_path = os.path.join('./emojis', args.data_y)
            test_path = os.path.join('./emojis', 'Test_{}'.format(args.data_y))
            self.train_dataset.append(datasets.ImageFolder(train_path, transform))
            self.test_dataset.append(datasets.ImageFolder(test_path, transform))

    def train_dataloader(self):
        dataloader = []
        for data in self.train_dataset:
            dataloader.append(DataLoader(dataset=data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True))
        return dataloader

    def val_dataloader(self):
        dataloader = []
        for data in self.test_dataset:
            dataloader.append(DataLoader(dataset=data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True))
        return dataloader

    def test_dataloader(self):
        dataloader = []
        for data in self.test_dataset:
            dataloader.append(DataLoader(dataset=data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True))
        return dataloader