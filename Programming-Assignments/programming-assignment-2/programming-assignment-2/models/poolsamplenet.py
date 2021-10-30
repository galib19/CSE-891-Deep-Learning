# poolsamplenet.py

import torch.nn as nn

__all__ = ["PoolUpsampleNet"]

class PoolUpsampleNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super().__init__()

        # Useful parameters
        padding = kernel // 2

        ############### YOUR CODE GOES HERE ###############
        self.module1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.module2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.module3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())

        self.module4 = nn.Sequential(
            nn.Conv2d(num_filters, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.module_last = nn.Conv2d(num_colours, num_colours, kernel_size=kernel, padding=padding)
        ###################################################        

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        ###################################################
        out = self.module1(x)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module_last(out)
        return out