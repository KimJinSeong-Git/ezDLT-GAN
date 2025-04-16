import torch
import torch.nn as nn

class DCGANLoss:
    def __init__(self, device='cpu'):
        self.criterion = nn.BCELoss()
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
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp

    def d_loss(self, real_pred, fake_pred, real_data, fake_data, discriminator):
        batch_size = real_data.size(0)
        device = real_data.device

        # Gradient Penalty
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        interpolated_pred = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

        wasserstein_loss = fake_pred.mean() - real_pred.mean()
        return wasserstein_loss + gradient_penalty

    def g_loss(self, fake_pred):
        return -fake_pred.mean()
