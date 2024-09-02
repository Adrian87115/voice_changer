import torch
import torch.nn as nn
import torch.nn.functional as f

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.downsample1 = self._block(self.in_channels, (3, 9), 32, (1, 1))
        self.downsample2 = self._block(32, (3, 8), 32, (1, 2))
        self.downsample3 = self._block(32, (3, 8), 32, (1, 2))
        self.downsample4 = self._block(32, (3, 6), 32, (1, 2))
        self.conv = nn.Conv2d(in_channels = 32,
                              out_channels = self.out_channels,
                              kernel_size = (36, 5),
                              stride = (1, 2),
                              padding = 0)
        self.sigmoid = nn.Sigmoid()


@staticmethod
def _block(in_channels, kernel_size, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = kernel_size,
                  stride = stride,
                  padding = 0),
        nn.BatchNorm2d(out_channels),
        nn.GLU(dim = 1))

def forward(self, x):
    down1 = self.downsample1(x)
    down2 = self.downsample2(down1)
    down3 = self.downsample3(down2)
    down4 = self.downsample4(down3)
    conv = self.conv(down4)
    sig = self.sigmoid(conv)
    product = torch.prod(sig, dim = 1)
    return product