from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Module


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            bias=False,
        )
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.ins_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.instance_norm:
            x = self.ins_norm(x)
        x = self.activation(x)
        return x


class Discriminator(Module):
    """
    Discriminator for 64x64 RGB images, matching the notebook architecture.
    """

    def __init__(self, conv_dim: int = 32) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.conv1 = ConvBlock(in_channels=3, out_channels=conv_dim, kernel_size=4)
        self.conv2 = ConvBlock(conv_dim, conv_dim * 2, kernel_size=4)
        self.conv3 = ConvBlock(conv_dim * 2, conv_dim * 4, kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_dim * 4 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, 1, 1, 1)
        return x


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class Generator(Module):
    """
    DCGAN-style generator, matching the notebook architecture.
    """

    def __init__(self, latent_dim: int, conv_dim: int = 32) -> None:
        super().__init__()
        self.deconv1 = DeconvBlock(
            in_channels=latent_dim,
            out_channels=conv_dim * 8,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.deconv2 = DeconvBlock(conv_dim * 8, conv_dim * 4, 4, 2, 1)
        self.deconv3 = DeconvBlock(conv_dim * 4, conv_dim * 2, 4, 2, 1)
        self.deconv4 = DeconvBlock(conv_dim * 2, conv_dim, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(conv_dim, 3, kernel_size=4, stride=2, padding=1)
        self.last_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.last_activation(x)
        return x

