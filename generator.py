import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, conv_dim = 32, num_speakers = 12):
        super(Generator, self).__init__()
        self.num_speakers = num_speakers
        self.downsample1 = self._down_block(1, (3, 9), conv_dim * 2, (1, 1))
        self.downsample2 = self._down_block(conv_dim, (4, 8), conv_dim * 4, (2, 2))
        self.downsample3 = self._down_block(conv_dim * 2, (4, 8), conv_dim * 8, (2, 2))
        self.downsample4 = self._down_block(conv_dim * 4, (3, 5), conv_dim * 4, (1, 1))
        self.downsample5 = self._down_block(conv_dim * 2, (9, 5), 10, (9, 1))
        self.upsample4 = self._up_block(5 + self.num_speakers, (9, 5), conv_dim * 2, (9, 1))
        self.upsample3 = self._up_block(conv_dim * 2 + self.num_speakers, (3, 5), conv_dim * 4, (1, 1))
        self.upsample2 = self._up_block(conv_dim * 4 + self.num_speakers, (4, 8), conv_dim * 2, (2, 2))
        self.upsample1 = self._up_block(conv_dim * 2 + self.num_speakers, (4, 8), conv_dim, (2, 2))
        self.deconv = nn.ConvTranspose2d(in_channels = conv_dim + self.num_speakers,
                                         out_channels = 1,
                                         kernel_size = (3, 11),
                                         stride = (1, 1),
                                         padding = (0, 2))

    @staticmethod
    def _down_block(in_channels, kernel_size, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 2),
            nn.BatchNorm2d(out_channels, affine = True, track_running_stats = True),
            nn.GLU(dim = 1))

    @staticmethod
    def _up_block(in_channels, kernel_size, out_channels, stride):
        out_channels = out_channels * 2
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 2),
            nn.BatchNorm2d(out_channels, affine = True, track_running_stats = True),
            nn.GLU(dim = 1))

    def forward(self, x, c):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        down5 = self.downsample5(down4)

        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, down5.size(2), down5.size(3))
        down5 = torch.cat([down5, c1], dim = 1)
        up4 = self.upsample4(down5)

        c2 = c.repeat(1, 1, up4.size(2), up4.size(3))
        up4 = torch.cat([up4, c2], dim = 1)
        up3 = self.upsample3(up4)

        c3 = c.repeat(1, 1, up3.size(2), up3.size(3))
        up3 = torch.cat([up3, c3], dim = 1)
        up2 = self.upsample2(up3)

        c4 = c.repeat(1, 1, up2.size(2), up2.size(3))
        up2 = torch.cat([up2, c4], dim=1)
        up1 = self.upsample1(up2)

        c5 = c.repeat(1, 1, up1.size(2), up1.size(3))
        up1 = torch.cat([up1, c5], dim=1)
        deconv = self.deconv(up1)
        return deconv