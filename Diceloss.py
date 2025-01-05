import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
# MSE_loss = torch.nn.SmoothL1Loss(reduction='mean')

def dice_loss(score, target):
    b, c, h, w = score.size()
    target = (target - target.min()) / (target.max()-target.min() + 1e-8)
    score = (score - score.min()) / (score.max()-score.min() + 1e-8)
    # s = score.view(b, c, -1)
    # # gt = target.view(b, c, -1)
    # s = F.softmax(s, dim=-1)
    # # gt = F.softmax(gt, dim=-1)
    # score = s.view(b, c, h, w)
    # s_ = F.sigmoid(score)
    # gt_ = F.sigmoid(target)

    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

# def mse_loss(score, target):
#     b, c, h, w = score.size()
#     # target = (target - target.min()) / (target.max()-target.min() + 1e-8)
#     # score = (score - score.min()) / (score.max()-score.min() + 1e-8)
#     mse = MSE_loss(score, target)
#     return mse
# def dice_loss(score, target):
#     b, c, h, w = score.size()
#     target = (target - target.min()) / (target.max()-target.min() + 1e-8)
#     score = (score - score.min()) / (score.max()-score.min() + 1e-8)
#     # s = score.view(b, c, -1)
#     # # gt = target.view(b, c, -1)
#     # s = F.softmax(s, dim=-1)
#     # # gt = F.softmax(gt, dim=-1)
#     # score = s.view(b, c, h, w)
#     # s_ = F.sigmoid(score)
#     # gt_ = F.sigmoid(target)
#
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss + MSE_loss(score, target)


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss