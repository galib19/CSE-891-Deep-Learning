# unet.py

import torch.nn as nn
import torch

__all__ = ["UNet"]

class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super().__init__()

        # Useful parameters
        stride = 2
        padding = kernel // 2
        output_padding = 1

        ############### YOUR CODE GOES HERE ############### 
        self.module1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, stride=stride, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.module2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, stride=stride, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.module3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, stride=stride, kernel_size=kernel, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())

        self.module4 = nn.Sequential(
            nn.ConvTranspose2d(num_filters+num_filters, num_colours, kernel_size=kernel, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.module_last = nn.Conv2d(num_colours+num_in_channels, num_colours, kernel_size=kernel, padding=padding)
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        ###################################################
        out1 = self.module1(x)
        out2 = self.module2(out1)
        out3 = self.module3(out2)
        out4 = self.module4(torch.cat((out3, out1), dim =1))
        out_final = self.module_last(torch.cat((out4, x), dim = 1))

        return out_final