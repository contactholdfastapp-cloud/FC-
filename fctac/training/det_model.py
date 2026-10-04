"""Tiny CenterNet-style point detector (training side, PyTorch).

Outputs at stride 4:
  hm   (3 ch, sigmoid): player foot point, ball centre, controlled player foot
  off  (2 ch): sub-pixel offset of the peak
  hgt  (1 ch): log(player height in input pixels)

~0.3 M parameters, depthwise-separable, export-friendly (Conv/BN/ReLU6/
Upsample only) so ONNX Runtime CUDA/DirectML/TensorRT and CPU all run it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HM = 3     # player, ball, controlled


def _bn_act(c):
    return nn.Sequential(nn.BatchNorm2d(c), nn.ReLU6(inplace=True))


class DS(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.b1 = _bn_act(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.b2 = _bn_act(cout)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.b2(self.pw(self.b1(self.dw(x))))
        return x + y if self.res else y


class TinyCenterNet(nn.Module):
    def __init__(self, width: float = 1.0):
        super().__init__()
        c = [max(8, int(v * width)) for v in (16, 24, 48, 96, 64, 48)]
        self.stem = nn.Sequential(nn.Conv2d(3, c[0], 3, 2, 1, bias=False), _bn_act(c[0]))
        self.s4 = nn.Sequential(DS(c[0], c[1], 2), DS(c[1], c[1]))
        self.s8 = nn.Sequential(DS(c[1], c[2], 2), DS(c[2], c[2]))
        self.s16 = nn.Sequential(DS(c[2], c[3], 2), DS(c[3], c[3]), DS(c[3], c[3]))
        self.lat16 = nn.Conv2d(c[3], c[4], 1)
        self.lat8 = nn.Conv2d(c[2], c[4], 1)
        self.fuse8 = DS(c[4], c[4])
        self.lat4 = nn.Conv2d(c[1], c[4], 1)
        self.fuse4 = DS(c[4], c[5])
        self.head = nn.Sequential(nn.Conv2d(c[5], c[5], 3, 1, 1), nn.ReLU6(inplace=True))
        self.hm = nn.Conv2d(c[5], N_HM, 1)
        self.off = nn.Conv2d(c[5], 2, 1)
        self.hgt = nn.Conv2d(c[5], 1, 1)
        nn.init.constant_(self.hm.bias, -4.0)       # rare positives: start near p=0.02

    def forward(self, x):
        x = self.stem(x)
        f4 = self.s4(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        p8 = self.fuse8(self.lat8(f8) + F.interpolate(self.lat16(f16), scale_factor=2.0, mode="nearest"))
        p4 = self.fuse4(self.lat4(f4) + F.interpolate(p8, scale_factor=2.0, mode="nearest"))
        h = self.head(p4)
        return torch.sigmoid(self.hm(h)), self.off(h), self.hgt(h)


def focal_loss(pred, gt, eps=1e-6):
    """CenterNet penalty-reduced focal loss on gaussian heatmaps."""
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pred = pred.clamp(eps, 1 - eps)
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n
