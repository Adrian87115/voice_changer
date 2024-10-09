import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, conv_dim = 32, num_speakers = 4):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers
        self.downsample1 = self._block(1 + self.num_speakers, (3, 9), conv_dim * 2, (1, 1))
        self.downsample2 = self._block(conv_dim + self.num_speakers, (3, 8), conv_dim * 2, (1, 2))
        self.downsample3 = self._block(conv_dim + self.num_speakers, (3, 8), conv_dim * 2, (1, 2))
        self.downsample4 = self._block(conv_dim + self.num_speakers, (3, 6), 2, (1, 2))
        self.conv = nn.Conv2d(in_channels = 1 + self.num_speakers,
                              out_channels = 1,
                              kernel_size = (36, 5),
                              stride = (36, 1),
                              padding = 1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _block(in_channels, kernel_size, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 2),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim = 1))

    def forward(self, x, c):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        down1 = self.downsample1(x)

        c2 = c.repeat(1, 1, down1.size(2), down1.size(3))
        down1 = torch.cat([down1, c2], dim=1)
        down2 = self.downsample2(down1)

        c3 = c.repeat(1, 1, down2.size(2), down2.size(3))
        down2 = torch.cat([down2, c3], dim=1)
        down3 = self.downsample3(down2)

        c4 = c.repeat(1, 1, down3.size(2), down3.size(3))
        down3 = torch.cat([down3, c4], dim=1)
        down4 = self.downsample4(down3)

        c5 = c.repeat(1, 1, down4.size(2), down4.size(3))
        down4 = torch.cat([down4, c5], dim=1)
        conv = self.conv(down4)

        sig = self.sigmoid(conv)
        product = torch.mean(sig, dim = [1, 2, 3], keepdim = False)
        return product