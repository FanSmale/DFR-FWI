# -*- coding: utf-8 -*-
"""
@Time : 2025/11/25 10:57

@Author : Zeng Zifei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class DeformableResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DeformableResBlock, self).__init__()
        padding = kernel_size // 2

        self.offset_conv1 = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                      kernel_size=3, padding=1, bias=True)
        self.deform_conv1 = DeformConv2d(in_channels, out_channels,
                                         kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.offset_conv2 = nn.Conv2d(out_channels, 2 * kernel_size * kernel_size,
                                      kernel_size=3, padding=1, bias=True)
        self.deform_conv2 = DeformConv2d(out_channels, out_channels,
                                         kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        offset1 = self.offset_conv1(x)
        out = F.relu(self.bn1(self.deform_conv1(x, offset1)))

        offset2 = self.offset_conv2(out)
        out = self.bn2(self.deform_conv2(out, offset2))

        out += identity
        out = F.relu(out)
        return out


class LargeKernelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(LargeKernelResBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class DWLargeKernelResBlock(nn.Module):

    def __init__(self, channels, kernel_size=7):
        super(DWLargeKernelResBlock, self).__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x

        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.pw_conv(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        # Assuming the input from each block is concatenated along the channel dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoublePath_Large(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, route_type='A'):
        super(DoublePath_Large, self).__init__()

        self.local_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            ResBlock(mid_channels, mid_channels)
        )
        self.global_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            DWLargeKernelResBlock(mid_channels)
        )

        if route_type == 'A':
            self.route_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        elif route_type == 'B':
            self.route_gate = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels // 4, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels // 4, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.route_gate = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.fusion = FusionBlock(2 * mid_channels, out_channels)

    def forward(self, x):
        feat_conv = self.local_path(x)  # (B, mid_channels, H, W)
        feat_Large = self.global_path(x)  # (B, mid_channels, H, W)

        alpha = self.route_gate(x)
        gate_conv = feat_conv * alpha
        gate_Large = feat_Large * (1.0 - alpha)

        out = self.fusion(gate_conv, gate_Large)

        return out


class DoublePath_Deform(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, route_type='A'):
        super(DoublePath_Deform, self).__init__()

        self.local_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            ResBlock(mid_channels, mid_channels)
        )
        self.global_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            DeformableResBlock(mid_channels, mid_channels)
        )

        if route_type == 'A':
            self.route_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        elif route_type == 'B':
            self.route_gate = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels // 4, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels // 4, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.route_gate = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.fusion = FusionBlock(2 * mid_channels, out_channels)

    def forward(self, x):
        feat_conv = self.local_path(x)  # (B, mid_channels, H, W)
        feat_deform = self.global_path(x)  # (B, mid_channels, H, W)

        alpha = self.route_gate(x)
        gate_conv = feat_conv * alpha
        gate_deform = feat_deform * (1.0 - alpha)

        out = self.fusion(gate_conv, gate_deform)

        return out