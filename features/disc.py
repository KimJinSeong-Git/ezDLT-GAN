from torch import nn
import math

class DCGAN_discriminator(nn.Module):
    def __init__(self, input_channels=3, input_size=64):
        super(DCGAN_discriminator, self).__init__()
        assert input_size in [2 ** i for i in range(5, 11)], "input_size must be a power of 2 between 32 and 1024"

        min_maps = 128
        max_channels = 1024
        init_res = 4
        nb_downsample = int(math.log2(input_size // init_res))

        layers = []

        in_channels = input_channels
        out_channels = min(min_maps, max_channels)

        # Step 1: Initial conv (no BatchNorm)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = out_channels

        # Step 2: Downsampling blocks
        for i in range(1, nb_downsample):
            out_channels = min(in_channels * 2, max_channels)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # Final 4x4 → 1x1
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        out = self.discriminator(x)  # (B, 1, 1, 1)
        return out.view(-1)