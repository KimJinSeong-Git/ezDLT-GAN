from features import gen, disc, loss
from torch import nn
    
class GAN_Model(nn.Module):
    def __init__(self, device, z_dim, model_name, img_shape):
        super(GAN_Model, self).__init__()
        if model_name == "DCGAN":
            img_channels = img_shape[0]
            img_size = img_shape[1]
            self.generator = gen.DCGAN_generator(z_dim=z_dim, output_channels=img_channels, output_size=img_size)
            self.discriminator = disc.DCGAN_discriminator(input_channels=img_channels, input_size=img_size)
            self.loss = loss.DCGANLoss(device)

        if model_name == "WGAN":
            img_channels = img_shape[0]
            img_size = img_shape[1]
            
            self.generator = gen.DCGAN_generator(z_dim=z_dim, output_channels=img_channels, output_size=img_size)
            self.discriminator = disc.DCGAN_discriminator(input_channels=img_channels, input_size=img_size)
            self.loss = loss.DCGANLoss(device)