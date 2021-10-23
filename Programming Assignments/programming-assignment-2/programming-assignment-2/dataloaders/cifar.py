# oxford.py

import os
import sys
import wget
import torch
import pickle
import tarfile
import numpy as np
import torch.nn as nn
import numpy.random as npr
import pytorch_lightning as pl
from torchvision import io, transforms, datasets
from torch.utils.data import DataLoader, Dataset

__all__ = ['CIFARColorizationData']

class MyCIFAR10(Dataset):
    def __init__(self, files_path, colors_path, train=False, cat=7):
        self.cat = cat
        self.train = train
        self.files_path = files_path
        colors = np.load(colors_path, allow_pickle=True, encoding="bytes")[0]

        self.num_colors = colors.shape[0]
        x, y = self.load_cifar10(train=train)
        x, y = self.process(x, y)
        x = self.get_rgb_cat(x, colors)
        self.x = y
        self.y = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = (torch.from_numpy(self.x[index]).float() - 0.5) / 0.5
        y = torch.from_numpy(self.y[index])
        return x, y

    def get_rgb_cat(self, xs, colours):
        """
        Get colour categories given RGB values. This function doesn't
        actually do the work, instead it splits the work into smaller
        chunks that can fit into memory, and calls helper function
        _get_rgb_cat

        Args:
        xs: float numpy array of RGB images in [B, C, H, W] format
        colours: numpy array of colour categories and their RGB values
        Returns:
        result: int numpy array of shape [B, 1, H, W]
        """
        if np.shape(xs)[0] < 100:
            return self._get_rgb_cat(xs)
        batch_size = 100
        nexts = []
        for i in range(0, np.shape(xs)[0], batch_size):
            next = self._get_rgb_cat(xs[i : i + batch_size, :, :, :], colours)
            nexts.append(next)
        result = np.concatenate(nexts, axis=0)
        return result


    def _get_rgb_cat(self, xs, colours):
        """
        Get colour categories given RGB values. This is done by choosing
        the colour in `colours` that is the closest (in RGB space) to
        each point in the image `xs`. This function is a little memory
        intensive, and so the size of `xs` should not be too large.

        Args:
        xs: float numpy array of RGB images in [B, C, H, W] format
        colours: numpy array of colour categories and their RGB values
        Returns:
        result: int numpy array of shape [B, 1, H, W]
        """
        num_colours = np.shape(colours)[0]
        xs = np.expand_dims(xs, 0)
        cs = np.reshape(colours, [num_colours, 1, 3, 1, 1])
        dists = np.linalg.norm(xs - cs, axis=2)  # 2 = colour axis
        cat = np.argmin(dists, axis=0)
        cat = np.expand_dims(cat, axis=1)
        return cat
    
    def process(self, xs, ys, max_pixel=256.0, downsize_input=False):
        """
        Pre-process CIFAR10 images by taking only the horse category,
        shuffling, and have colour values be bound between 0 and 1

        Args:
        xs: the colour RGB pixel values
        ys: the category labels
        max_pixel: maximum pixel value in the original data
        Returns:
        xs: value normalized and shuffled colour images
        grey: greyscale images, also normalized so values are between 0 and 1
        """
        xs = xs / max_pixel
        xs = xs[np.where(ys == self.cat)[0], :, :, :]
        npr.shuffle(xs)

        grey = np.mean(xs, axis=1, keepdims=True)

        if downsize_input:
            downsize_module = nn.Sequential(
                nn.AvgPool2d(2),
                nn.AvgPool2d(2),
                nn.Upsample(scale_factor=2),
                nn.Upsample(scale_factor=2),
            )
            xs_downsized = downsize_module.forward(torch.from_numpy(xs).float())
            xs_downsized = xs_downsized.data.numpy()
            return (xs, xs_downsized)
        else:
            return (xs, grey)
        

    def get_file(self, fname, origin, cache_dir="data"):
        datadir = os.path.join(cache_dir)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        fpath = os.path.join(datadir, fname)

        print("File path: %s" % fpath)
        if not os.path.exists(fpath):
            print("Downloading data from", origin)
            filename = wget.download(origin, out=cache_dir)
            print("Extracting file.")
            with tarfile.open(filename) as archive:
                archive.extractall(datadir)

    def load_batch(self, fpath, label_key="labels"):
        """Internal utility for parsing CIFAR data.
        # Arguments
            fpath: path the file to parse.
            label_key: key for label data in the retrieve
                dictionary.
        # Returns
            A tuple `(data, labels)`.
        """
        f = open(fpath, "rb")
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding="bytes")
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode("utf8")] = v
            d = d_decoded
        f.close()
        data = d["data"]
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels


    def load_cifar10(self, train=True, transpose=False):
        """Loads CIFAR10 dataset.
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        cache_dir = 'data'
        dirname = "cifar-10-batches-py"
        filename = "cifar-10-python.tar.gz"
        origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.get_file(filename, origin=origin)
        path = os.path.join(cache_dir, dirname)

        if train:
            num_train_samples = 50000
            x = np.zeros((num_train_samples, 3, 32, 32), dtype="uint8")
            y = np.zeros((num_train_samples,), dtype="uint8")

            for i in range(1, 6):
                fpath = os.path.join(path, "data_batch_" + str(i))
                data, labels = self.load_batch(fpath)
                x[(i - 1) * 10000 : i * 10000, :, :, :] = data
                y[(i - 1) * 10000 : i * 10000] = np.array(labels)
            y = np.reshape(y, (len(y), 1))
            if transpose:
                x = x.transpose(0, 2, 3, 1)
        else:
            fpath = os.path.join(path, "test_batch")
            x, y = self.load_batch(fpath)
            y = np.reshape(y, (len(y), 1))
            if transpose:
                x = x.transpose(0, 2, 3, 1)
        
        return (x, y)

class CIFARColorizationData(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self, split='train'):
        if split == 'train':            
            self.dataset_val = MyCIFAR10(self.opts.data_dir, self.opts.colors, train=False)
            self.dataset_train = MyCIFAR10(self.opts.data_dir, self.opts.colors, train=True)
        else:
            self.dataset_test = MyCIFAR10(self.opts.data_dir, self.opts.colors, train=False)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.opts.batch_size,
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