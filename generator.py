import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_channels = 1
        self.out_channels = 1

        self.downsample1 = self._down_block(self.in_channels, (3, 9), 32, (1, 1))
        self.downsample2 = self._down_block(32, (4, 8), 64, (2, 2))
        self.downsample3 = self._down_block(64, (4, 8), 128, (2, 2))
        self.downsample4 = self._down_block(128, (3, 5), 64, (1, 1))
        self.downsample5 = self._down_block(64, (9, 5), 5, (9, 1))

        self.upsample4 = self._up_block(5, (9, 5), 64, (9, 1))
        self.upsample3 = self._up_block(64, (3, 5), 128, (1, 1))
        self.upsample2 = self._up_block(128, (4, 8), 64, (2, 2))
        self.upsample1 = self._up_block(64, (4, 8), 32, (2, 2))

        self.deconv = nn.ConvTranspose2d(in_channels = 32,
                                          out_channels = self.out_channels,
                                          kernel_size = (3, 9),
                                          stride = 1,
                                          padding = 0)

    @staticmethod
    def _down_block(in_channels, kernel_size, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim = 1))

    @staticmethod
    def _up_block(in_channels, kernel_size, out_channels, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim = 1))

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        down1 = self.downsample1(x)
        print(f"Down1 shape: {down1.shape}")
        down2 = self.downsample2(down1)
        print(f"Down2 shape: {down2.shape}")
        down3 = self.downsample3(down2)
        print(f"Down3 shape: {down3.shape}")
        down4 = self.downsample4(down3)
        print(f"Down4 shape: {down4.shape}")
        down5 = self.downsample5(down4)
        print(f"Down5 shape: {down5.shape}")
        up4 = self.upsample4(down5)
        print(f"Up4 shape: {up4.shape}")
        up3 = self.upsample3(up4)
        print(f"Up3 shape: {up3.shape}")
        up2 = self.upsample2(up3)
        print(f"Up2 shape: {up2.shape}")
        up1 = self.upsample1(up2)
        print(f"Up1 shape: {up1.shape}")
        return up1
