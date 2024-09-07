import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                  nn.InstanceNorm2d(out_channels, affine = True, track_running_stats = True),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                  nn.InstanceNorm2d(out_channels, affine = True, track_running_stats = True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, num_speakers, conv_dim = 32):
        super(Generator, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.num_speakers = num_speakers
        self.downsample1 = self._down_block(self.in_channels + self.num_speakers, (3, 9), conv_dim, (1, 1))
        self.downsample2 = self._down_block(conv_dim, (4, 8), conv_dim * 2, (2, 2))
        self.downsample3 = self._down_block(conv_dim * 2, (4, 8), conv_dim * 4, (2, 2))
        self.downsample4 = self._down_block(conv_dim * 4, (3, 5), conv_dim * 2, (1, 1))
        self.downsample5 = self._down_block(conv_dim * 2, (9, 5), 5, (9, 1))

        self.residual_blocks = nn.Sequential(*[ResidualBlock(5, 5) for _ in range(6)])

        self.upsample4 = self._up_block(5, (9, 5), conv_dim * 2, (9, 1))
        self.upsample3 = self._up_block(conv_dim * 2, (3, 5), conv_dim * 4, (1, 1))
        self.upsample2 = self._up_block(conv_dim * 4, (4, 8), conv_dim * 2, (2, 2))
        self.upsample1 = self._up_block(conv_dim * 2, (4, 8), conv_dim, (2, 2))

        self.deconv = nn.ConvTranspose2d(in_channels = conv_dim,
                                         out_channels = self.out_channels,
                                         kernel_size = (9, 9),
                                         stride = (1, 1),
                                         padding = (0,0))

    @staticmethod
    def _down_block(in_channels, kernel_size, out_channels, stride):
        out_channels = out_channels * 2 #GLU halves the number of channels
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 1),
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
                      padding = 1),
            nn.BatchNorm2d(out_channels, affine = True, track_running_stats = True),
            nn.GLU(dim = 1))

    def forward(self, x, speaker_embedding):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = torch.cat([x, speaker_embedding], dim = 1)
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        down5 = self.downsample5(down4)
        res = self.residual_blocks(down5)
        up4 = self.upsample4(res)
        up3 = self.upsample3(up4)
        up2 = self.upsample2(up3)
        up1 = self.upsample1(up2)
        deconv = self.deconv(up1)
        return deconv