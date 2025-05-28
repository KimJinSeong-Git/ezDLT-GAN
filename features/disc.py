import torch
import torch.nn as nn
import torch.nn.init as init

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.discriminator.add_module("conv_final", nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        output = self.discriminator(x).view(-1)
        return output
    
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)
        M_T = M.permute(1, 0, 2, 3)
        norm = torch.abs(M - M_T).sum(3)
        norm = torch.clamp(norm, min=1e-6, max=50)
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)

        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class MinibatchDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64, input_size=64, nb_classes=10):
        super(MinibatchDiscriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        output_size = input_size // (2 ** 4)
        feat_dim = feature_maps * 8 * output_size * output_size
            
        self.mbd = MinibatchDiscrimination(feat_dim, 64, 50)
        self.fc = nn.Linear(feat_dim+64, nb_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        feat = self.feature_extractor(x).view(batch_size, -1)
        mbd_out = self.mbd(feat)
        output = self.fc(mbd_out)
        return feat, output
    