import torch
import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, num_target_speakers, conv_dim = 8):
        super(DomainClassifier, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.num_target_speakers = num_target_speakers
        self.downsample1 = self._block(self.in_channels, (4, 4), conv_dim, (2, 2))
        self.downsample2 = self._block(conv_dim, (4, 4), conv_dim * 2, (2, 2))
        self.downsample3 = self._block(conv_dim * 2, (4, 4), conv_dim * 4, (2, 2))
        self.downsample4 = self._block(conv_dim * 4, (3, 4), conv_dim * 2, (1, 2))
        self.conv = nn.Conv2d(in_channels = conv_dim * 2,
                              out_channels = self.num_target_speakers,
                              kernel_size = (1, 4),
                              stride = (1, 2),
                              padding = 0)
        self.softmax = nn.Softmax(dim = 1)


    @staticmethod
    def _block(in_channels, kernel_size, out_channels, stride):
        out_channels = out_channels * 2
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim = 1))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        conv = self.conv(down4)
        softmax = self.softmax(conv)
        prod = torch.mean(softmax, dim = [2, 3], keepdim = False)
        return prod