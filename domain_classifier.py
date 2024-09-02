import torch
import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.downsample1 = self._block(self.in_channels, (4, 4), 8, (2, 2))
        self.downsample2 = self._block(8, (4, 4), 16, (2, 2))
        self.downsample3 = self._block(16, (4, 4), 32, (2, 2))
        self.downsample4 = self._block(32, (3, 4), 16, (1, 2))
        self.conv = nn.Conv2d(in_channels = 16,
                              out_channels = 4,
                              kernel_size = (1, 4),
                              stride = (1, 2),
                              padding = 0)
        self.softmax = nn.Softmax(dim = 1)


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
    softmax = self.softmax(conv)
    product = torch.prod(softmax)
    return product