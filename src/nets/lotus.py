import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import functools
import torchvision 
from torchvision import transforms as T
import pytorch_lightning as pl

import torchmetrics


from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from monai import transforms

import numpy as np

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=16, spatial_size=32):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x


class UltrasoundRendering(pl.LightningModule):
    def __init__(self, num_labels=340, alpha_coeff_boundary_map=0.1, beta_coeff_scattering=0.1, tgc=8, clamp_vals=False):
        super().__init__()               
        self.save_hyperparameters()
        
        self.acoustic_impedance_dict = torch.nn.Parameter(torch.rand(self.hparams.num_labels))    # Z in MRayl
        self.attenuation_dict =    torch.nn.Parameter(torch.rand(self.hparams.num_labels))   # alpha in dB cm^-1 at 1 MHz
        self.mu_0_dict =           torch.nn.Parameter(torch.rand(self.hparams.num_labels)) # mu_0 - scattering_mu   mean brightness
        self.mu_1_dict =           torch.nn.Parameter(torch.rand(self.hparams.num_labels)) # mu_1 - scattering density, Nr of scatterers/voxel
        self.sigma_0_dict =        torch.nn.Parameter(torch.rand(self.hparams.num_labels)) # sigma_0 - scattering_sigma - brightness std
        g_kernel = torch.tensor(self.gaussian_kernel(3, 0., 0.5)[None, None, :, :])

        self.register_buffer("g_kernel", g_kernel)

        self.resize_t = T.Resize((256, 256))

        self.maxAngle = torch.nn.Parameter(data=torch.tensor(60.0 / 2 / 180 * np.pi))        

        self.minAngle = torch.nn.Parameter(data=torch.tensor(-60.0 / 2 / 180 * np.pi))

        self.minRadius = torch.nn.Parameter(data=torch.tensor(140.0))
        self.maxRadius = torch.nn.Parameter(data=torch.tensor(340.0))
        
    def init_params(self, df):

        # accoustic_imped,attenuation,mu_0,mu_1,sigma_0
        self.acoustic_impedance_dict = torch.nn.Parameter(torch.tensor(df['accoustic_imped']))    # Z in MRayl
        self.attenuation_dict =    torch.nn.Parameter(torch.tensor(df['attenuation']))   # alpha in dB cm^-1 at 1 MHz
        self.mu_0_dict =           torch.nn.Parameter(torch.tensor(df['mu_0'])) # mu_0 - scattering_mu   mean brightness
        self.mu_1_dict =           torch.nn.Parameter(torch.tensor(df['mu_1'])) # mu_1 - scattering density, Nr of scatterers/voxel
        self.sigma_0_dict =        torch.nn.Parameter(torch.tensor(df['sigma_0'])) # sigma_0 - scattering_sigma - brightness std

    def gaussian_kernel(self, size: int, mean: float, std: float):
        d1 = torch.distributions.Normal(mean, std)
        d2 = torch.distributions.Normal(mean, std*3)
        vals_x = d1.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()
        vals_y = d2.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()

        gauss_kernel = torch.einsum('i,j->ij', vals_x, vals_y)
        
        return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)
    
    def rendering(self, shape, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, z_vals=None, refl_map=None, boundary_map=None):
        
        dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
        dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
        dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                # dists.shape=(W, H)

        attenuation = torch.exp(-attenuation_medium_map * dists)
        attenuation_total = torch.cumprod(attenuation, dim=3, dtype=torch.float32, out=None)

        gain_coeffs = torch.linspace(1, self.hparams.tgc, attenuation_total.shape[3], device=self.device)
        gain_coeffs = torch.tile(gain_coeffs, (attenuation_total.shape[2], 1))
        gain_coeffs = torch.tensor(gain_coeffs)
        attenuation_total = attenuation_total * gain_coeffs     # apply TGC

        reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=3, dtype=torch.float32, out=None) 
        reflection_total = reflection_total.squeeze(-1) 
        reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

        texture_noise = torch.randn(shape, dtype=torch.float32, device=self.device)
        scattering_probability = torch.randn(shape, dtype=torch.float32, device=self.device)        

        z = mu_1_map - scattering_probability
        sigmoid_map = torch.sigmoid(self.hparams.beta_coeff_scattering * z)

        # scattering_zero = torch.zeros(shape, dtype=torch.float32)
        # approximating  Eq. (4) to be differentiable:
        # where(scattering_probability <= mu_1_map, 
        #                     texture_noise * sigma_0_map + mu_0_map, 
        #                     scattering_zero)
        # scatterers_map =  (sigmoid_map) * (texture_noise * sigma_0_map + mu_0_map) + (1 -sigmoid_map) * scattering_zero   # Eq. (6)
        scatterers_map =  (sigmoid_map) * (texture_noise * sigma_0_map + mu_0_map)

        psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map, weight=self.g_kernel, stride=1, padding="same")
        # psf_scatter_conv = psf_scatter_conv.squeeze()

        b = attenuation_total * psf_scatter_conv    # Eq. (3)

        border_convolution = torch.nn.functional.conv2d(input=boundary_map, weight=self.g_kernel, stride=1, padding="same")
        # border_convolution = border_convolution.squeeze()

        r = attenuation_total * reflection_total * refl_map * border_convolution # Eq. (2)
        
        intensity_map = b + r   # Eq. (1)
        # intensity_map = intensity_map.squeeze() 
        intensity_map = torch.clamp(intensity_map, 0, 1)

        return intensity_map, attenuation_total, reflection_total_plot, scatterers_map, scattering_probability, border_convolution, texture_noise, b, r
    
    def render_rays(self, W, H):
        N_rays = W 
        t_vals = torch.linspace(0., 1., H, device=self.device)
        z_vals = t_vals.unsqueeze(0).expand(N_rays , -1) * 4 

        return z_vals

    # warp the linear US image to approximate US image from curvilinear US probe 
    def warp_img(self, inputImage):
        resultWidth = 360
        resultHeight = 220
        centerX = resultWidth / 2
        centerY = -120.0
        # maxAngle =  60.0 / 2 / 180 * np.pi #rad
        # minAngle = -maxAngle
        # minRadius = 140.0
        # maxRadius = 340.0
        
        h = inputImage.shape[2]
        w = inputImage.shape[3]

        # Create x and y grids
        x = torch.arange(resultWidth, device=self.device).float() - centerX
        y = torch.arange(resultHeight, device=self.device).float() - centerY
        xx, yy = torch.meshgrid(x, y)

        # Calculate angle and radius
        angle = torch.atan2(xx, yy)
        radius = torch.sqrt(xx ** 2 + yy ** 2)

        # Create masks for angle and radius
        angle_mask = (angle > self.minAngle) & (angle < self.maxAngle)
        radius_mask = (radius > self.minRadius) & (radius < self.maxRadius)

        # Calculate original column and row
        origCol = (angle - self.minAngle) / (self.maxAngle - self.minAngle) * w
        origRow = (radius - self.minRadius) / (self.maxRadius - self.minRadius) * h

        # Reshape input image to be a batch of 1 image
        inputImage = inputImage.float()

        # Scale original column and row to be in the range [-1, 1]
        origCol = origCol / (w - 1) * 2 - 1
        origRow = origRow / (h - 1) * 2 - 1

        # Transpose input image to have channels first
        inputImage = torch.transpose(inputImage, 2, 3)

        # Use grid_sample to interpolate
        repeats = [1,]*len(inputImage.shape)
        repeats[0] = inputImage.shape[0]
        grid = torch.stack([origCol, origRow], dim=-1).repeat(repeats).to(self.device)
        resultImage = F.grid_sample(inputImage, grid, mode='bilinear', align_corners=True)

        # Apply masks and set values outside of mask to 0
        mask = angle_mask & radius_mask
        mask = mask.repeat(repeats)
        resultImage[~(mask)] = 0.0
        
        resultImage_resized = self.resize_t(resultImage)

        return resultImage_resized


    def forward(self, x):
        #init tissue maps
        #generate maps from the dictionary and the input label map
        x = torch.rot90(x, k=1, dims=[2, 3])
        acoustic_imped_map = self.acoustic_impedance_dict[x]
        attenuation_medium_map = self.attenuation_dict[x]
        mu_0_map = self.mu_0_dict[x]
        mu_1_map = self.mu_1_dict[x]
        sigma_0_map = self.sigma_0_dict[x]

        
        #Comput the difference along dimension 2
        diff_arr = torch.diff(acoustic_imped_map, dim=2)                
        # The pad tuple is (padding_left,padding_right, padding_top,padding_bottom)
        # The array is padded at the top
        diff_arr = F.pad(diff_arr, (0,0,1,0))

        #Compute the boundary map using the diff_array
        boundary_map =  -torch.exp(-(diff_arr**2)/self.hparams.alpha_coeff_boundary_map) + 1
        
        #Roll/shift the elements along dimension 2 and set the last element to 0
        shifted_arr = torch.roll(acoustic_imped_map, -1, dims=2)
        shifted_arr[-1:] = 0

        # This computes the sum/accumulation along the direction and set elements that are 0 to 1. Compute the division
        sum_arr = acoustic_imped_map + shifted_arr
        sum_arr[sum_arr == 0] = 1
        div = diff_arr / sum_arr
        # Compute the reflection from the elements
        refl_map = div ** 2
        refl_map = torch.sigmoid(refl_map)      # 1 / (1 + (-refl_map).exp())

        z_vals = self.render_rays(x.shape[2], x.shape[3])

        if self.hparams.clamp_vals:
            attenuation_medium_map = torch.clamp(attenuation_medium_map, 0, 10)
            acoustic_imped_map = torch.clamp(acoustic_imped_map, 0, 10)
            sigma_0_map = torch.clamp(sigma_0_map, 0, 1)
            mu_1_map = torch.clamp(mu_1_map, 0, 1)
            mu_0_map = torch.clamp(mu_0_map, 0, 1)

        ret_list = self.rendering(x.shape, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, z_vals=z_vals, refl_map=refl_map, boundary_map=boundary_map)

        intensity_map  = ret_list[0]
       
        # return intensity_map
        intensity_map_masked = self.warp_img(intensity_map)        
        intensity_map_masked = torch.rot90(intensity_map_masked, k=3, dims=[2, 3])
        
        return intensity_map_masked,  attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, acoustic_imped_map, boundary_map, shifted_arr, intensity_map


class RealUS(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator = UltrasoundRendering(num_labels=self.hparams.num_labels, alpha_coeff_boundary_map=self.hparams.alpha_coeff_boundary_map, beta_coeff_scattering=self.hparams.beta_coeff_scattering, tgc=self.hparams.tgc, clamp_vals=self.hparams.clamp_vals)        
        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
        
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False
    
    def get_us(self, x):
        return self.generator(x)[0]

    def configure_optimizers(self):        
        
        optimizer_g = optim.AdamW(self.generator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def compute_loss_g(self, real, fake):
        """Calculate the loss for generator G"""

        perceptual_loss = self.perceptual_loss(fake.float(), real.float()) * self.hparams.parceptual_weight
        # combined loss and calculate gradients

        if self.trainer.current_epoch > self.hparams.warm_up_n_epochs:
            # GAN loss D_A(G_A(A))
            logits_fake = self.discriminator(fake.contiguous().float())[-1]
            adv_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)*self.hparams.adversarial_weight

            return adv_loss + perceptual_loss

        return perceptual_loss

    def compute_loss_d(self, discriminator, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        logits_real = discriminator(real.contiguous().float())[-1]        
        loss_discriminator_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)        
        # Fake
        logits_fake = discriminator(fake.detach().contiguous().float())[-1]        
        loss_discriminator_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)        
        # Combined loss and calculate gradients
        return (loss_discriminator_real + loss_discriminator_fake) * 0.5

    def training_step(self, train_batch, batch_idx):
        labeled, real_us = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        # forward        
        fake_us = self.generator(labeled)[0]  # G_A(A)

        # Optimize generator
        self.set_requires_grad([self.discriminator], requires_grad=False)
        optimizer_g.zero_grad()        
        loss_g = self.compute_loss_g(real_us, fake_us)

        self.log("train_loss_g", loss_g)
        
        loss_g.backward()
        optimizer_g.step()
      
        # D_A and D_B
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.warm_up_n_epochs:
            

            self.set_requires_grad([self.discriminator], requires_grad=True)
            optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
            
            loss_d = self.compute_loss_d(self.discriminator, real_us, fake_us)      # calculate gradients for D_b
            loss_d.backward()
            
            optimizer_d.step()  # update D_A and D_B's weights

            self.log("train_loss_d", loss_d)

        return loss_g + loss_d
        
    
    def validation_step(self, val_batch, batch_idx):        
        labeled, real_us = val_batch

        fake_us = self.generator(labeled)[0]  # G_A(A)        
        val_loss = self.compute_loss_g(real_us, fake_us)
        
        # loss_d = self.compute_loss_d(self.discriminator, real_a, fake_a)

        # val_loss = 
        

        self.log("val_loss", val_loss, sync_dist=True)

class RealUSKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator = UltrasoundRendering(num_labels=self.hparams.num_labels, alpha_coeff_boundary_map=self.hparams.alpha_coeff_boundary_map, beta_coeff_scattering=self.hparams.beta_coeff_scattering, tgc=self.hparams.tgc, clamp_vals=self.hparams.clamp_vals)
        self.autoencoder = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=8,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )
        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
        
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False
    def get_us(self, x):
        return self.autoencoder(self.generator(x)[0])[0]

    def configure_optimizers(self):
        
        g_params = list(self.generator.parameters()) + list(self.autoencoder.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def compute_loss_g(self, real, fake):
        """Calculate the loss for generator G"""

        perceptual_loss = self.perceptual_loss(fake.float(), real.float()) * self.hparams.parceptual_weight
        # combined loss and calculate gradients

        if self.trainer.current_epoch > self.hparams.warm_up_n_epochs:
            # GAN loss D_A(G_A(A))
            logits_fake = self.discriminator(fake.contiguous().float())[-1]
            adv_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)*self.hparams.adversarial_weight

            return adv_loss + perceptual_loss

        return perceptual_loss

    def compute_loss_d(self, discriminator, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        logits_real = discriminator(real.contiguous().float())[-1]        
        loss_discriminator_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)        
        # Fake
        logits_fake = discriminator(fake.detach().contiguous().float())[-1]        
        loss_discriminator_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)        
        # Combined loss and calculate gradients
        return (loss_discriminator_real + loss_discriminator_fake) * 0.5

    def training_step(self, train_batch, batch_idx):
        labeled, real_us = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        # forward        
        fake_us = self.generator(labeled)[0]  # G_A(A)
        fake_us, z_mu, z_sigma = self.autoencoder(fake_us)

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Optimize generator
        self.set_requires_grad([self.discriminator], requires_grad=False)
        optimizer_g.zero_grad()        
        loss_g = self.compute_loss_g(real_us, fake_us)

        self.log("train_loss_g", loss_g)
        
        loss_g.backward()
        optimizer_g.step()
      
        # D_A and D_B
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.warm_up_n_epochs:
            

            self.set_requires_grad([self.discriminator], requires_grad=True)
            optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
            
            loss_d = self.compute_loss_d(self.discriminator, real_us, fake_us)      # calculate gradients for D_b
            loss_d.backward()
            
            optimizer_d.step()  # update D_A and D_B's weights

            self.log("train_loss_d", loss_d)

        return loss_g + loss_d
        
    
    def validation_step(self, val_batch, batch_idx):        
        labeled, real_us = val_batch

        fake_us = self.generator(labeled)[0]  # G_A(A)
        fake_us, z_mu, z_sigma = self.autoencoder(fake_us)
        
        val_loss = self.compute_loss_g(real_us, fake_us)
        
        # loss_d = self.compute_loss_d(self.discriminator, real_a, fake_a)

        # val_loss = 
        

        self.log("val_loss", val_loss, sync_dist=True)