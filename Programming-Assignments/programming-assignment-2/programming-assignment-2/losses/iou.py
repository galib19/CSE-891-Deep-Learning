# iou.py

from torch import nn

class SoftIoU(nn.Module):
    def __init__(self):
        super(SoftIoU, self).__init__()
        self.SMOOTH=1e-6

    def __call__(self, pred, gt):
        # Compute the IoU between the pred and the gt (ground truth)
        ################ YOUR CODE GOES HERE ######################
        # Around 5 lines of code
        # Hint:
        # - apply softmax on pred along the channel dimension (dim=1)
        # - only have to compute IoU between gt and the foreground channel of pred
        # - no need to consider IoU for the background channel of pred
        # - extract foreground from the softmaxed pred (e.g., softmaxed_pred[:, 1, :, :])
        # - compute intersection between foreground and gt
        # - compute union between foreground and gt
        # - compute loss using the computed intersection and union
        ######################################################
        loss = 0
        return loss