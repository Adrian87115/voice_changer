import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 128,
                               kernel_size = (3, 3),
                               stride = (1, 1),
                               padding = (1, 1))
        self.glu = nn.GLU(dim = 1)
        self.down1 = self._block(in_channels = 128 // 2,
                                 out_channels = 256,
                                 kernel_size = (3, 3),
                                 stride = (2, 2),
                                 padding = (1, 1))
        self.down2 = self._block(in_channels = 256 // 2,
                                 out_channels = 512,
                                 kernel_size = (3, 3),
                                 stride = (2, 2),
                                 padding = (1, 1))
        self.down3 = self._block(in_channels = 512 // 2,
                                 out_channels = 1024,
                                 kernel_size = (3, 3),
                                 stride = (2, 2),
                                 padding = (1, 1))
        self.down4 = self._block(in_channels = 1024 // 2,
                                 out_channels = 1024,
                                 kernel_size = (1, 5),
                                 stride = (1, 1),
                                 padding = (0, 2))
        self.conv2 = nn.Conv2d(in_channels = 1024 // 2,
                               out_channels = 1,
                               kernel_size = (1, 3),
                               stride = (1, 1),
                               padding = (0, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = padding),
            nn.InstanceNorm2d(num_features = out_channels, affine = True),
            nn.GLU(dim = 1))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        conv1 = self.conv1(x)
        glu = self.glu(conv1)
        down1 = self.down1(glu)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        conv2 = self.conv2(down4)
        output = self.sigmoid(conv2)
        return output