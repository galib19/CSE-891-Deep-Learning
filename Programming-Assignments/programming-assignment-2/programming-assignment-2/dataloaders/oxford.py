# oxford.py

import os
import cv2
import wget
import torch
import tarfile
import numpy as np
from scipy.io import loadmat
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

__all__ = ['OxfordFlowersData']

class CUB(Dataset):
    def __init__(self, files_path, train=True):

        if not os.path.exists(os.path.join(files_path, "17flowers.tgz")):
            print("Downloading flower dataset")
            url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
            filename = wget.download(url, out=files_path)
            with tarfile.open(filename, "r") as f:
                f.extractall(files_path)
        if not os.path.exists(os.path.join(files_path, "trimaps.tgz")):
            url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz'
            filename = wget.download(url, out=files_path)
            with tarfile.open(filename, "r") as f:
                f.extractall(files_path)
        if not os.path.exists(os.path.join(files_path, "datasplits.mat")):
            url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat'
            filename = wget.download(url, out=files_path)

        self.files_path = files_path
        self.split = loadmat(os.path.join(files_path, "datasplits.mat"))
        if train:
            filenames = (
                list(self.split["trn1"][0])
                + list(self.split["trn2"][0])
                + list(self.split["trn3"][0])
            )
        else:
            # We only use `val1` for validation
            filenames = self.split["val1"][0]

        valid_filenames = []
        for i in filenames:
            img_name = "image_%04d.jpg" % int(i)
            if os.path.exists(os.path.join(files_path, "jpg", img_name)) and os.path.exists(
                os.path.join(files_path, "trimaps",
                             img_name.replace("jpg", "png"))
            ):
                valid_filenames.append(img_name)

        self.valid_filenames = valid_filenames
        self.num_files = len(valid_filenames)

    def normalize(self, im):
        """Normalizes images with Imagenet stats."""
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return (im / 255.0 - imagenet_stats[0]) / imagenet_stats[1]


    def denormalize(self, img):
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return img * imagenet_stats[1] + imagenet_stats[0]

    def read_image(self, path):
        im = cv2.imread(str(path))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def convert_to_binary(self, mask, thres=0.5):
        binary_masks = (
            (mask[0] == 128) & (mask[1] == 0) & (mask[2] == 0)
        ) + 0.0
        return binary_masks

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):

        filename = self.valid_filenames[index]

        # Load the image
        path = os.path.join(self.files_path, "jpg", filename)
        x = self.read_image(path)  # H*W*c
        x = cv2.resize(x, (224, 224))
        x = self.normalize(x)
        x = np.rollaxis(x, 2)  # To meet torch's input specification(c*H*W)
        x = torch.from_numpy(x).float()

        # Load the segmentation mask
        path = os.path.join(self.files_path, "trimaps",
                            filename.replace("jpg", "png"))
        y = self.read_image(path)
        y = cv2.resize(y, (224, 224))  # H*W*c
        y = np.rollaxis(y, 2)  # To meet torch's input specification(c*H*W)
        y = self.convert_to_binary(y)
        y = torch.from_numpy(y).long()

        return x, y


class OxfordFlowersData(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self, split='train'):
        if split == 'train':            
            self.dataset_val = CUB(files_path='data/', train=False)
            self.dataset_train = CUB(files_path='data', train=True)
        else:
            self.dataset_test = CUB(files_path='data', train=False)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.opts.batch_size_val,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        return dataloader