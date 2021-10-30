# deeplabv3.py

import torch
import torch.nn as nn

__all__ = ["DeepLabV3"]

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet101', pretrained=True)

        # We only learn the last layer and freeze all the other weights
        ################ Code goes here ######################
        # Around 3-4 lines of code
        # Hint:
        # - freeze the gradient of all layers (2 lines)
        # - replace the classifier.4 layer with a new Conv2d layer (1-2 lines)
        ######################################################
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        # self.model.requires_grad_(False)
        self.model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1), stride=(1,1))

    def forward(self, x):
        return self.model(x)
