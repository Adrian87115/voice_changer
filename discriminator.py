import torch
import torch.nn as nn
import torch.nn.functional as f

class Discriminator(nn.Module):
    def __init__(self, conv_dim = 32, num_speakers = 12):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers
        self.in_channels = 1
        self.out_channels = 1
        self.downsample1 = self._block(self.in_channels, (3, 9), conv_dim, (1, 1))
        self.downsample2 = self._block(conv_dim, (3, 8), conv_dim, (1, 2))
        self.downsample3 = self._block(conv_dim, (3, 8), conv_dim, (1, 2))
        self.downsample4 = self._block(conv_dim, (3, 6), conv_dim, (1, 2))
        self.conv = nn.Conv2d(in_channels = conv_dim,
                              out_channels = self.out_channels,
                              kernel_size = (36, 5),
                              stride = (36, 1),
                              padding = 1)
        self.sigmoid = nn.Sigmoid()
        self.projection = nn.Linear(num_speakers, conv_dim)

    @staticmethod
    def _block(in_channels, kernel_size, out_channels, stride):
        out_channels = out_channels * 2
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
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        speaker_proj = self.projection(c).unsqueeze(2).unsqueeze(3)
        conditioned_features = down4 + speaker_proj
        conv = self.conv(conditioned_features)
        sig = self.sigmoid(conv)
        product = torch.mean(sig, dim = [1, 2, 3], keepdim = False)
        return product