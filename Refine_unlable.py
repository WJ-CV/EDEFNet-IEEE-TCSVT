import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

class HA(nn.Module):
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)

        Soft_Att= soft_attention.max(attention)
        zero = torch.zeros_like(Soft_Att)
        one = torch.ones_like(Soft_Att)

        Soft_Att = torch.tensor(torch.where(Soft_Att > 0.05, one, Soft_Att))
        Soft_Att = torch.tensor(torch.where(Soft_Att <=0.05, zero, Soft_Att))

        refine_unsal = torch.mul(x, Soft_Att)

        return refine_unsal#, Depth_neg

class Refine_unlable(nn.Module):
    def __init__(self):
        super(Refine_unlable, self).__init__()
        self.HA = HA()
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, pred_sal, init_lable):
        refine_unsal = self.HA(pred_sal.sigmoid(),init_lable)

        res = pred_sal + (pred_sal - (1 - refine_unsal.sigmoid()))
        zero = torch.zeros_like(res)
        depth_s = torch.tensor(torch.where(res <= 0.0, zero, res)).sigmoid()
        refine_semantic = min_max_norm(depth_s)

        return refine_semantic

