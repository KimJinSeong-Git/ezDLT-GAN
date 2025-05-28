from features import gen, disc
import torch.nn as nn

class WGAN(nn.Module):
    def __init__(self, img_size, z_size, nb_feat_maps=128):
        super(WGAN, self).__init__()

        self.generator = gen.Generator(z_size, 3, nb_feat_maps, img_size)
        self.discriminator = disc.Discriminator(3, nb_feat_maps)

class InversionGAN(nn.Module):
    def __init__(self, img_size, z_size, device, nb_feat_maps=128, nb_classes=10):
        super(InversionGAN, self).__init__()
        self.generator = gen.Generator(z_size, 3, nb_feat_maps, img_size).to(device)
        self.discriminator = disc.MinibatchDiscriminator(nb_classes=nb_classes+1).to(device)