import torch
import torch.nn as nn

class ConditionalInstanceNormalisation(nn.Module):
    def __init__(self, in_channels, num_speakers):
        super(ConditionalInstanceNormalisation, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.gamma_t = nn.Linear(num_speakers, in_channels)
        self.beta_t = nn.Linear(num_speakers, in_channels)

    def forward(self, x, c_trg):
        u = torch.mean(x, dim = 2, keepdim = True)
        var = torch.mean((x - u) * (x - u), dim = 2, keepdim = True)
        std = torch.sqrt(var + 1e-8)
        gamma = self.gamma_t(c_trg.to(self.device))
        gamma = gamma.view(-1, self.in_channels, 1, 1)
        beta = self.beta_t(c_trg.to(self.device))
        beta = beta.view(-1, self.in_channels, 1, 1)
        h = (x - u) / std
        h = h * gamma + beta
        return h

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_speakers):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.cin1 = ConditionalInstanceNormalisation(out_channels, num_speakers)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.cin2 = ConditionalInstanceNormalisation(out_channels, num_speakers)

    def forward(self, x, c):
        out = self.conv1(x)
        out = self.cin1(out, c)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.cin2(out, c)
        return self.relu(out + x)

class Generator(nn.Module):
    def __init__(self, conv_dim = 32, num_speakers = 12):
        super(Generator, self).__init__()
        self.num_speakers = num_speakers
        self.in_channels = 1
        self.out_channels = 1
        self.downsample1 = self._down_block(self.in_channels, (3, 9), conv_dim, (1, 1))
        self.downsample2 = self._down_block(conv_dim, (4, 8), conv_dim * 2, (2, 2))
        self.downsample3 = self._down_block(conv_dim * 2, (4, 8), conv_dim * 4, (2, 2))
        self.downsample4 = self._down_block(conv_dim * 4, (3, 5), conv_dim * 2, (1, 1))
        self.downsample5 = self._down_block(conv_dim * 2, (9, 5), 5, (9, 1))

        self.residual_blocks = nn.Sequential(*[ResidualBlock(5, 5, self.num_speakers) for _ in range(6)])

        self.upsample4 = self._up_block(5, (9, 5), conv_dim * 2, (9, 1))
        self.upsample3 = self._up_block(conv_dim * 2, (3, 5), conv_dim * 4, (1, 1))
        self.upsample2 = self._up_block(conv_dim * 4, (4, 8), conv_dim * 2, (2, 2))
        self.upsample1 = self._up_block(conv_dim * 2, (4, 8), conv_dim, (2, 2))

        self.deconv = nn.ConvTranspose2d(in_channels = conv_dim,
                                         out_channels = self.out_channels,
                                         kernel_size = (3, 11),
                                         stride = (1, 1),
                                         padding = (0, 2))

    @staticmethod
    def _down_block(in_channels, kernel_size, out_channels, stride):
        out_channels = out_channels * 2 #GLU halves the number of channels
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

    def forward(self, x, c_trg):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        down5 = self.downsample5(down4)
        res = down5
        for block in self.residual_blocks:
            res = block(res, c_trg)
        up4 = self.upsample4(res)
        up3 = self.upsample3(up4)
        up2 = self.upsample2(up3)
        up1 = self.upsample1(up2)
        deconv = self.deconv(up1)
        return deconv