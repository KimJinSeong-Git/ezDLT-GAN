from torch import nn
import math

class DCGAN_generator(nn.Module):
    def __init__(self, z_dim=128, output_channels=3, output_size=64):
        super(DCGAN_generator, self).__init__()
        assert output_size in [2 ** i for i in range(5, 11)], "output_size must be a power of 2 between 32 and 1024"

        min_maps = 128        
        max_channels = 1024

        init_res = 4  # initial 4x4 resolution
        nb_upsample = int(math.log2(output_size // init_res))

        layers = []

        # Step 1: z → (feature_maps * 8) x 4 x 4
        in_channels = z_dim
        out_channels = min(min_maps * (2 ** nb_upsample), max_channels)        
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))

        in_channels = out_channels

        # Step 2: Upsample blocks
        for i in range(nb_upsample):
            is_last = (i == nb_upsample - 1)
            out_channels = max(in_channels // 2, min_maps)

            if is_last:
                layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
                in_channels = out_channels

        self.generator = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.generator(z)