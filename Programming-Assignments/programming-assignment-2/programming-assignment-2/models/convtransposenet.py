# convtransposenet.py

import torch.nn as nn

__all__ = ["ConvTransposeNet"]

class ConvTransposeNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super().__init__()

        # Useful parameters
        stride = 2
        padding = kernel // 2
        output_padding = 1

        ############### YOUR CODE GOES HERE ############### 
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        ###################################################
        pass