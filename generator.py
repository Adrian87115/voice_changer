import torch
import torch.nn as nn
import torch.nn.functional as f

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()
        self.residualLayer = nn.Sequential(nn.Conv1d(in_channels = in_channels,
                                                     out_channels = out_channels,
                                                     kernel_size = kernel_size,
                                                     stride = stride,
                                                     padding = padding),
                                           nn.InstanceNorm1d(num_features = out_channels, affine = True),
                                           nn.GLU(dim = 1),
                                           nn.Conv1d(in_channels = out_channels // 2,
                                                     out_channels = in_channels,
                                                     kernel_size = kernel_size,
                                                     stride = stride,
                                                     padding = padding),
                                           nn.InstanceNorm1d(num_features = in_channels, affine = True))

    def forward(self, input):
        out = self.residualLayer(input)
        return input + out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 128,
                               kernel_size = (5, 15),
                               stride = (1, 1),
                               padding = (2, 7))
        self.glu = nn.GLU(dim = 1)
        self.down1 = self._down_block(in_channels = 128 // 2,
                                      out_channels = 256,
                                      kernel_size = 5,
                                      stride = (2, 2),
                                      padding = (2, 2))
        self.down2 = self._down_block(in_channels = 256 // 2,
                                      out_channels = 512,
                                      kernel_size = (5, 5),
                                      stride = (2, 2),
                                      padding = (2, 2))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels = 2304,
                                             out_channels = 256,
                                             kernel_size = 1,
                                             padding = 0),
                                   nn.InstanceNorm1d(num_features = 256, affine = True))
        self.residual1 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.residual2 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.residual3 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.residual4 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.residual5 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.residual6 = ResidualLayer(in_channels = 256,
                                       out_channels = 512,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels = 256,
                                             out_channels = 2304,
                                             kernel_size = 1,
                                             stride = 1,
                                             padding = 0),
                                   nn.InstanceNorm1d(num_features = 2304, affine = True))
        self.up1 = self._up_block(in_channels = 512, # 256
                                  out_channels = 1024,
                                  kernel_size = (5, 5),
                                  stride = (3, 1),
                                  padding = (2, 2))
        self.up2 = self._up_block(in_channels = 256 // 2,
                                  out_channels = 512,
                                  kernel_size = (5, 5),
                                  stride = (3, 1),
                                  padding = (2, 2))
        self.conv4 = nn.Conv2d(in_channels = 128 // 2,
                               out_channels = 35,
                               kernel_size = (5, 15),
                               stride = (3, 1), # (4, 1)
                               padding = (2, 7))

    @staticmethod
    def _down_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                       out_channels = out_channels,
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding = padding),
                             nn.InstanceNorm2d(num_features = out_channels, affine = True),
                             nn.GLU(dim = 1))

    @staticmethod
    def _up_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                       out_channels = out_channels,
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding = padding),
                             nn.PixelShuffle(upscale_factor = 2),
                             nn.InstanceNorm2d(num_features = out_channels // 4, affine = True),
                             nn.GLU(dim = 1))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1) # x is of shape b, c, h, w = 1, 1, 35, 128
        conv1 = self.conv1(x)
        glu = self.glu(conv1)
        down1 = self.down1(glu)
        down2 = self.down2(down1)
        down3 = down2.view([down2.shape[0], 2304, -1])
        down3 = self.conv2(down3)
        residual1 = self.residual1(down3)
        residual2 = self.residual2(residual1)
        residual3 = self.residual3(residual2)
        residual4 = self.residual4(residual3)
        residual5 = self.residual5(residual4)
        residual6 = self.residual6(residual5)
        residual6 = self.conv3(residual6)
        residual6 = residual6.view([x.shape[0], 512, 9, -1]) # 256
        up1 = self.up1(residual6)
        up2 = self.up2(up1)
        conv4 = self.conv4(up2)
        output = conv4.view([x.shape[0], 1, 35, -1]).squeeze(1)
        return output