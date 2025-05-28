import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib.pyplot as plt
import json, os
from tqdm import tqdm

from features import loss

class WGAN_Traininer:
    def __init__(self, device, gan_model, train_loader, img_size, z_size, learning_rate, save_dir):
        self.device = device
        self.gan_model = gan_model.to(device)
        self.train_loader = train_loader

        self.img_size = img_size
        self.z_size = z_size
        self.learning_rate = learning_rate

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(device)
        
        self.d_optim = optim.Adam(self.gan_model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.g_optim = optim.Adam(self.gan_model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        self.criterion = loss.WGANGPLoss(device, z_size)
        self.save_dir = save_dir

        self.fixed_z = torch.randn(16, z_size, device=self.device)

    def train_step(self):
        total_d_loss = 0
        total_g_loss = 0

        for x_real, y_real in tqdm(self.train_loader, desc="Training step"):
            # setup
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)
            bs = y_real.size(0)

            z = torch.randn(bs, self.z_size).to(self.device)

            # discriminator
            self.d_optim.zero_grad()
            x_fake = self.gan_model.generator(z)
            d_out_real = self.gan_model.discriminator(x_real)
            d_out_fake = self.gan_model.discriminator(x_fake)

            d_loss = self.criterion.get_d_loss(bs, self.gan_model.discriminator, d_out_real, d_out_fake, x_real, x_fake)
            d_loss.backward()
            self.d_optim.step()
            
            # generator
            self.g_optim.zero_grad()
            g_loss = self.criterion.get_g_loss(bs, self.gan_model.discriminator, self.gan_model.generator, self.feature_extractor)
            g_loss.backward()
            self.g_optim.step()

            # add loss
            total_d_loss += d_loss.detach().cpu().item()
            total_g_loss += g_loss.detach().cpu().item()

        total_d_loss /= len(self.train_loader)
        total_g_loss /= len(self.train_loader)

        return total_d_loss, total_g_loss
    
    def save_log(self, log):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title(f'Discriminator Loss')
        axes[0].set_xlabel(f'Epochs')
        axes[0].set_ylabel(f'Loss')
        axes[0].plot(log['d_loss'])

        axes[1].set_title(f'Generator Loss')
        axes[1].set_xlabel(f'Epochs')
        axes[1].set_ylabel(f'Loss')
        axes[1].plot(log['g_loss'])

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training log.png', format='png')

        with open(f'{self.save_dir}/training log.json', 'w') as json_file:
            json.dump(log, json_file, indent=4)

    def show_generated_samples(self):
        with torch.no_grad():
            fake_imgs = self.gan_model.generator(self.fixed_z).detach().cpu()

        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        fake_imgs_denorm = fake_imgs * std + mean

        grid = vutils.make_grid(fake_imgs_denorm, nrow=4, value_range=(-1, 1))

        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0))  # [C, H, W] → [H, W, C]
        plt.show()

    def run(self, epochs):
        log = {
            "d_loss": [],
            "g_loss": []
        }

        for epoch in range(epochs):
            d_loss, g_loss = self.train_step()

            print(f"Epoch [{epoch+1}/{epochs}] - Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")
            log["d_loss"].append(d_loss)
            log["g_loss"].append(g_loss)

            if epoch % 10 == 0:
                torch.save(self.gan_model.state_dict(), f"{self.save_dir}/CheckPoint_model.pth")
                #self.show_generated_samples()
                print(f"Check point model saved at epoch {epoch}")

        torch.save(self.gan_model.state_dict(), f"{self.save_dir}/Final_model.pth")
        self.save_log(log)
        print(f"Final model saved at epoch {epoch+1}")

class InversionGAN_Trainer:
    def __init__(self, device, gan_model, target_model, trainloader, nb_classes, z_size, learning_rate, save_dir):
        self.device = device
        self.gan_model = gan_model
        self.target_model = target_model

        self.trainloader = trainloader

        self.z_size = z_size
        self.nb_classes = nb_classes
        self.save_dir = save_dir

        self.criterion = loss.InversionGANLoss(nb_classes)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.d_optim = optim.Adam(self.gan_model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.g_optim = optim.Adam(self.gan_model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        self.fixed_z = torch.randn(16, z_size, device=self.device)

    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad_(False) 

    def unfreeze(self, model):
        for p in model.parameters():
            p.requires_grad_(True)

    def train_step(self, trainloader_unsup1, trainloader_unsup2):
        total_d_loss = 0
        total_g_loss = 0

        for x_real, y_real in tqdm(self.trainloader, desc="Training step"):
            bs = x_real.size(0)
            
            x_unsup1, _ = next(trainloader_unsup1)
            x_unsup2, _ = next(trainloader_unsup2)
            
            x_real = x_real.to(self.device)
            x_unsup1 = x_unsup1.to(self.device)
            x_unsup2 = x_unsup2.to(self.device)

            y_real = y_real.to(self.device)
            y_fake = torch.full((len(y_real),), self.nb_classes, device=self.device, dtype=torch.long)

            # discriminator
            self.freeze(self.gan_model.generator)
            self.unfreeze(self.gan_model.discriminator)

            self.d_optim.zero_grad()
            z1 = torch.randn((bs, self.z_size), device=self.device)
            x_fake1 = self.gan_model.generator(z1)

            out_real = self.target_model(x_real)
            
            _, d_out_real = self.gan_model.discriminator(x_real)
            _, d_out_fake1 = self.gan_model.discriminator(x_fake1)
            _, d_out_unsup1 = self.gan_model.discriminator(x_unsup1)

            d_loss = self.criterion.get_d_loss(self.cross_entropy, out_real, y_real, y_fake, d_out_real, d_out_fake1, d_out_unsup1)
            d_loss.backward()
            self.d_optim.step()
            
            # generator
            self.freeze(self.gan_model.discriminator)
            self.unfreeze(self.gan_model.generator)

            self.g_optim.zero_grad()
            z2 = torch.randn((bs, self.z_size), device=self.device)
            x_fake2 = self.gan_model.generator(z2)

            d_feat_fake2, d_out_fake2 = self.gan_model.discriminator(x_fake2)
            d_feat_unsup2, _ = self.gan_model.discriminator(x_unsup2)

            g_loss = self.criterion.get_g_loss(d_out_fake2, d_feat_fake2, d_feat_unsup2)
            g_loss.backward()
            self.g_optim.step()

            # add loss
            total_d_loss += d_loss.detach().cpu().item()
            total_g_loss += g_loss.detach().cpu().item()

        total_d_loss /= len(self.trainloader)
        total_g_loss /= len(self.trainloader)

        return total_d_loss, total_g_loss
    
    def save_log(self, log):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title(f'Discriminator Loss')
        axes[0].set_xlabel(f'Epochs')
        axes[0].set_ylabel(f'Loss')
        axes[0].plot(log['d_loss'])

        axes[1].set_title(f'Generator Loss')
        axes[1].set_xlabel(f'Epochs')
        axes[1].set_ylabel(f'Loss')
        axes[1].plot(log['g_loss'])

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training log.png', format='png')

        with open(f'{self.save_dir}/training log.json', 'w') as json_file:
            json.dump(log, json_file, indent=4)

    def show_generated_samples(self):
        with torch.no_grad():
            fake_imgs = self.gan_model.generator(self.fixed_z).detach().cpu()

        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        fake_imgs_denorm = fake_imgs * std + mean

        grid = vutils.make_grid(fake_imgs_denorm, nrow=4, value_range=(-1, 1))

        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0))  # [C, H, W] → [H, W, C]
        plt.show()

    def run(self, epochs, train_dataset, bs):
        log = {
            "d_loss": [],
            "g_loss": []
        }

        for epoch in range(epochs):
            trainloader_unsup1 = DataLoader(train_dataset, batch_size=bs, shuffle=True).__iter__()
            trainloader_unsup2 = DataLoader(train_dataset, batch_size=bs, shuffle=True).__iter__()
            d_loss, g_loss = self.train_step(trainloader_unsup1, trainloader_unsup2)

            print(f"Epoch [{epoch+1}/{epochs}] - Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")
            log["d_loss"].append(d_loss)
            log["g_loss"].append(g_loss)

            if epoch % 10 == 0:
                torch.save(self.gan_model.state_dict(), f"{self.save_dir}/CheckPoint_model.pth")
                #self.show_generated_samples()
                print(f"Check point model saved at epoch {epoch}")

        torch.save(self.gan_model.state_dict(), f"{self.save_dir}/Final_model.pth")
        self.save_log(log)
        print(f"Final model saved at epoch {epoch+1}")