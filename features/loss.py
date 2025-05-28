import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGANLoss:
    def __init__(self, device='cpu'):
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def d_loss(self, real_pred, fake_pred):
        real_labels = torch.ones_like(real_pred, device=self.device)
        fake_labels = torch.zeros_like(fake_pred, device=self.device)

        real_loss = self.criterion(real_pred, real_labels)
        fake_loss = self.criterion(fake_pred, fake_labels)
        return real_loss + fake_loss

    def g_loss(self, fake_pred):
        real_labels = torch.ones_like(fake_pred, device=self.device)
        return self.criterion(fake_pred, real_labels)

class WGANGPLoss:
    def __init__(self, device, z_size, lambda_gp=10):
        self.device = device
        self.z_size = z_size
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, bs, discriminator, x_real, x_fake):
        alpha = torch.rand(bs, 1, 1, 1, device=self.device)
        interpolated = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)

        interpolated_pred = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(bs, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty
    
    def diversity_loss(self, generator, feature_extractor, z1, z2):
        x_fake1, x_fake2 = generator(z1), generator(z2)
        feat_fake1, feat_fake2 = feature_extractor(x_fake1), feature_extractor(x_fake2)
        numerator = torch.norm(feat_fake1 - feat_fake2, p=2)
        denominator = torch.norm(z1 - z2, p=2) 
        return numerator / (denominator + 1e-8) 

    def get_d_loss(self, bs, discriminator, d_out_real, d_out_fake, x_real, x_fake):
        gp = self.gradient_penalty(bs, discriminator, x_real, x_fake)
        wasserstein_loss = torch.mean(d_out_fake) - torch.mean(d_out_real)

        return wasserstein_loss + self.lambda_gp*gp

    def get_g_loss(self, bs, discriminator, generator, feature_extractor, lambda_d=0.5):
        z1, z2 = torch.randn(bs, self.z_size).to(self.device), torch.randn(bs, self.z_size).to(self.device)
        x_fake = generator(z1)
        w_loss = -torch.mean(discriminator(x_fake))
        div_loss = self.diversity_loss(generator, feature_extractor, z1, z2)
        g_loss = w_loss + lambda_d*div_loss
        
        return g_loss
    
class InversionGANLoss:
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        
    def log_sum_exp(self, x, axis = 1):
        m = torch.max(x, dim = axis)[0]
        return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

    def priorLoss(self, d_out_unsup1, d_out_fake):
        real_softplus = torch.mean(F.softplus(self.log_sum_exp(d_out_unsup1)))
        real_logsumexp = torch.mean(self.log_sum_exp(d_out_unsup1))
        fake_softplus = torch.mean(F.softplus(self.log_sum_exp(d_out_fake)))

        loss = 0.5*(real_softplus-real_logsumexp+fake_softplus)

        return loss

    def softXEnt(self, d_out_real, out_real):
        targetprobs = nn.functional.softmax(d_out_real, dim = 1)
        logprobs = nn.functional.log_softmax(out_real, dim = 1)
        return -(targetprobs * logprobs).sum() / out_real.shape[0]

    def unsupLoss(self, criterion, y_real, d_out_real, y_fake, d_out_fake):
        loss_real = criterion(d_out_real, y_real)
        loss_fake = criterion(d_out_fake, y_fake)

        return 0.5 * (loss_real + loss_fake)
    
    def hLoss(self, d_out_fake):
        b = F.softmax(d_out_fake, dim=1)
        logb = F.log_softmax(d_out_fake, dim=1)
        b = -(b * logb).sum(dim=1).mean()

        return b
    
    def momLoss(self, d_feat_fake, d_feat_unsup2):
        mom_fake = torch.mean(d_feat_fake, dim = 0)
        mom_unsup2 = torch.mean(d_feat_unsup2, dim = 0)

        mom_loss = torch.mean((mom_fake - mom_unsup2).abs())

        return mom_loss
    
    def get_d_loss(self, criterion, out_real, y_real, y_fake, d_out_real, d_out_fake, d_out_unsup1):
        loss_reg = self.priorLoss(d_out_unsup1, d_out_fake)
        loss_sup = self.softXEnt(d_out_real[:, :self.nb_classes], out_real)
        loss_unsup = self.unsupLoss(criterion, y_real, d_out_real, y_fake, d_out_fake)

        loss = loss_reg + loss_sup + loss_unsup

        return loss
    
    def get_g_loss(self, d_out_fake, d_feat_fake, d_feat_unsup2, lamda=1e-4):
        mom_loss = self.momLoss(d_feat_fake, d_feat_unsup2)
        h_loss = self.hLoss(d_out_fake)
        prob = F.softmax(d_out_fake, dim=1)
        #adv_loss = -torch.log(prob[:, self.nb_classes - 1] + 1e-8).mean()

        loss = mom_loss + lamda * h_loss# + adv_loss
        return loss