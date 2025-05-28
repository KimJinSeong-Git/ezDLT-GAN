import torch
import torch.nn as nn
import torch.nn.init as init

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, feature_maps=64, output_size=64):
        super(Generator, self).__init__()

        self.output_size = output_size

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1)
        )

        self.generator.add_module("tanh", nn.Tanh())

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        output = self.generator(z)
        return output