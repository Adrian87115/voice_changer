
import torch
import torch.nn as nn
import torch.nn.functional as f

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class up2Dsample(nn.Module):
    def __init__(self, upscale_factor = 2):
        super(up2Dsample, self).__init__()
        self.scale_factor = upscale_factor

    def forward(self, input):
        h = input.shape[2]
        w = input.shape[3]
        new_size = [h * self.scale_factor, w * self.scale_factor]
        return f.interpolate(input, new_size)

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()
        self.residualLayer = nn.Sequential(nn.Conv1d(in_channels = in_channels,
                                                     out_channels = out_channels,
                                                     kernel_size = kernel_size,
                                                     stride = stride,
                                                     padding = padding),
                                           nn.InstanceNorm1d(num_features = out_channels, affine = True),
                                           GLU(),
                                           nn.Conv1d(in_channels = out_channels,
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
                               stride = 1,
                               padding = (2, 7))
        self.glu = GLU()
        self.down1 = self._down_block(in_channels = 128,
                                      out_channels = 256,
                                      kernel_size = 5,
                                      stride = 2,
                                      padding = 2)
        self.down2 = self._down_block(in_channels = 256,
                                      out_channels = 512,
                                      kernel_size = 5,
                                      stride = 2,
                                      padding = 2)
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels = 4608,
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
                                             out_channels = 4608,
                                             kernel_size = 1,
                                             stride = 1,
                                             padding = 0),
                                   nn.InstanceNorm1d(num_features = 4608, affine = True))
        self.up1 = self._up_block(in_channels = 512,
                                  out_channels = 1024,
                                  kernel_size = 5,
                                  stride = 1,
                                  padding = 2)
        self.up2 = self._up_block(in_channels = 1024,
                                  out_channels = 512,
                                  kernel_size = 5,
                                  stride = 1,
                                  padding = 2)
        self.conv4 = nn.Conv2d(in_channels = 512,
                               out_channels = 1,
                               kernel_size = (5, 15),
                               stride = 1,
                               padding = (2, 7))

    @staticmethod
    def _down_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                       out_channels = out_channels,
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding = padding),
                             nn.InstanceNorm2d(num_features = out_channels, affine = True),
                             GLU())

    @staticmethod
    def _up_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                       out_channels = out_channels,
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding = padding),
                             up2Dsample(upscale_factor = 2),
                             nn.InstanceNorm2d(num_features = out_channels, affine = True),
                             GLU())
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        conv1 = self.conv1(x)
        glu = self.glu(conv1)
        down1 = self.down1(glu)
        down2 = self.down2(down1)
        down3 = down2.view([down2.shape[0], -1, down2.shape[3]])
        down3 = self.conv2(down3)
        residual1 = self.residual1(down3)
        residual2 = self.residual2(residual1)
        residual3 = self.residual3(residual2)
        residual4 = self.residual4(residual3)
        residual5 = self.residual5(residual4)
        residual6 = self.residual6(residual5)
        residual6 = self.conv3(residual6)
        residual6 = residual6.view([down2.shape[0], down2.shape[1], down2.shape[2], down2.shape[3]])
        up1 = self.up1(residual6)
        up2 = self.up2(up1)
        output = self.conv4(up2)
        output = output.view([output.shape[0], -1, output.shape[3]])
        output = output[:, :35, :]
        return output