import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import functools
import torchvision 

import pytorch_lightning as pl

import torchmetrics


from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import LatentDiffusionInferer, DiffusionInferer

import monai
from monai import transforms

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

class AutoEncoderKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=latent_channels,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )

        # self.autoencoderkl = nets.AutoencoderKL(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 128, 256, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     attention_levels=(False, False, False, False),
        #     with_encoder_nonlocal_attn=False,
        #     with_decoder_nonlocal_attn=False,
        # )

        # self.autoencoderkl = nets.AutoencoderKL(spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 256, 512, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     norm_num_groups=32,
        #     attention_levels=(False, False, False, True))

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        if hasattr(self.hparams, "denoise") and self.hparams.denoise: 
            self.noise_transform = torch.nn.Sequential(
                GaussianNoise(0.0, 0.05),
                RandCoarseShuffle(),
                SaltAndPepper()
            )
        else:
            self.noise_transform = nn.Identity()

        if hasattr(self.hparams, "smooth") and self.hparams.smooth: 
            self.smooth_transform = transforms.RandSimulateLowResolution(prob=1.0, zoom_range=(0.15, 0.3))
        else:
            self.smooth_transform = nn.Identity()
        
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]
    
    def compute_loss_generator(self, x, reconstruction, z_mu, z_sigma):
        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_adv_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_adv_loss

        return loss_g, recons_loss
    
    def compute_loss_discriminator(self, x, reconstruction):
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        return (loss_d_fake + loss_d_real) * 0.5


    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.smooth_transform(self.noise_transform(x)))

        loss_g, recons_loss = self.compute_loss_generator(x, reconstruction, z_mu, z_sigma)

        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        loss_d = 0.0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            loss_d = self.compute_loss_discriminator(x, reconstruction)

            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        self.log("train_loss_recons", recons_loss)
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

    def forward(self, images):        
        return self.autoencoderkl(images)


class AutoEncoderKLPaired(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        y = train_batch[1]

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), y.float())
        p_loss = self.perceptual_loss(reconstruction.float(), y.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(y.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        y = val_batch[1]
        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        #     recon_loss = F.l1_loss(x.float(), reconstruction.float())

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(y.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.autoencoderkl(images)


class AutoEncoderTanh(nets.AutoencoderKL):
    def sampling(self, z_mu, z_sigma):
        x = super().sampling(z_mu, z_sigma)
        return F.tanh(x)

class AutoEncoderTanhPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 1

        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoder = AutoEncoderTanh(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoder.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        reconstruction, z_mu, z_sigma = self.autoencoder(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, z_mu, z_sigma = self.autoencoder(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images):        
        return self.autoencoder(images)


class DecoderKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.decoder = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        ).decoder

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
        )

        self.resize_transform = transforms.Resize([-1, 64, 64])
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.decoder.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        reconstruction = self(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        # kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)
        loss_g = recons_loss + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction = self(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images):        
        return self.decoder(self.resize_transform(images))

class AutoEncoderLowKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=self.hparams.latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
        )

        self.resize_transform_low = transforms.Resize([-1, 64, 64])
        self.resize_transform_high = transforms.Resize([-1, 256, 256])
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.model.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        reconstruction, z_mu, z_sigma = self(self.noise_transform(self.resize_transform_low(x)))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)
        loss_g = recons_loss + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, z_mu, z_sigma = self(self.resize_transform_low(x))
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images):        
        return self.model(self.resize_transform_high(images))

class CycleVQVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator_a = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(256, 512),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=32,
        )

        self.generator_b = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(256, 512),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=32,
        )

        self.discriminator_a = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        
        self.discriminator_b = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.cycle_loss = nn.L1Loss()
        self.idt_loss = nn.L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

    def get_b(self, x):
        return self.generator_b(x)[0]

    def get_a(self, x):
        return self.generator_a(x)[0]

    def configure_optimizers(self):
        g_params = list(self.generator_a.parameters()) + list(self.generator_b.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        d_params = list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters())
        optimizer_d = optim.AdamW(d_params,
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


    def compute_loss_g(self, real_a, fake_b, reconstruction_a, real_b, fake_a, reconstruction_b):
        """Calculate the loss for generators G_A and G_B"""
        # lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        # # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        logits_fake_b = self.discriminator_a(fake_b.contiguous().float())[-1]
        loss_generator_a = self.adv_loss(logits_fake_b, target_is_real=True, for_discriminator=False) * self.hparams.gamma_a
        # GAN loss D_B(G_B(B))        
        logits_fake_a = self.discriminator_b(fake_a.contiguous().float())[-1]
        loss_generator_b = self.adv_loss(logits_fake_a, target_is_real=True, for_discriminator=False)  * self.hparams.gamma_b
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_a = self.cycle_loss(reconstruction_a, real_a) * self.hparams.lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_b = self.cycle_loss(reconstruction_b, real_b) * self.hparams.lambda_b
        # combined loss and calculate gradients

        perceptual_loss_a = self.perceptual_loss(fake_a.float(), real_a.float()) * self.hparams.ommega_a
        perceptual_loss_b = self.perceptual_loss(fake_b.float(), real_b.float()) * self.hparams.ommega_b

        return loss_generator_a + loss_generator_b + loss_cycle_a + loss_cycle_b + perceptual_loss_a + perceptual_loss_b

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
        real_a, real_b = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        # forward        
        fake_b, q_loss_b = self.generator_a(real_a)  # G_A(A)
        reconstruction_a, q_loss_a = self.generator_b(fake_b)   # G_B(G_A(A))
        
        fake_a, q_loss_a = self.generator_b(real_b)  # G_B(B)
        reconstruction_b, q_loss_b = self.generator_a(fake_a)   # G_A(G_B(B))

        # Optimize generator
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], requires_grad=False)
        optimizer_g.zero_grad()        
        loss_g = self.compute_loss_g(real_a, fake_b, reconstruction_a, real_b, fake_a, reconstruction_b)
        
        loss_g.backward()
        optimizer_g.step()

      
        # D_A and D_B
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], requires_grad=True)
        optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
        loss_d_a = self.compute_loss_d(self.discriminator_b, real_a, fake_a)      # calculate gradients for D_b
        loss_d_a.backward()
        loss_d_b = self.compute_loss_d(self.discriminator_a, real_b, fake_b)      # calculate gradients for D_a
        loss_d_b.backward()
        optimizer_d.step()  # update D_A and D_B's weights

        loss_d = loss_d_a + loss_d_b

        # # Generator part
        # fake_us, quantization_loss_fake_us = self.get_a(x_must)
        # reconstruction_must, quantization_loss_must = self.get_b(fake_us)

        # fake_must, quantization_loss_fake_must = self.get_must(x_us)
        # reconstruction_us, quantization_loss_us = self.get_us(fake_must)

        # logits_fake_us = self.discriminator_us(fake_us.contiguous().float())[-1]
        # logits_fake_must = self.discriminator_must(fake_must.contiguous().float())[-1]

        # recons_loss_must = self.l1_loss(reconstruction_must.float(), x_must.float())
        # p_loss_must = self.perceptual_loss(fake_must.float(), x_must.float())

        # recons_loss_us = self.l1_loss(reconstruction_us.float(), x_us.float())
        # p_loss_us = self.perceptual_loss(fake_us.float(), x_us.float())
        
        
        # generator_loss = self.adv_loss(logits_fake_us, target_is_real=True, for_discriminator=False) + self.adv_loss(logits_fake_must, target_is_real=True, for_discriminator=False)
        # recons_loss = recons_loss_must + recons_loss_us
        # quantization_loss = quantization_loss_fake_us + quantization_loss_must + quantization_loss_fake_must + quantization_loss_us
        # p_loss = self.hparams.perceptual_weight * (p_loss_must + p_loss_us)

        # loss_g = recons_loss + quantization_loss + p_loss + self.hparams.adversarial_weight * generator_loss

        # loss_g.backward()
        # optimizer_g.step()

        # # Discriminator part
        # optimizer_d.zero_grad(set_to_none=True)

        # logits_fake_must = self.discriminator_must(reconstruction_must.contiguous().detach())[-1]
        # loss_d_fake_must = self.adv_loss(logits_fake_must, target_is_real=False, for_discriminator=True)
        # logits_real_must = self.discriminator_must(x_must.contiguous().detach())[-1]
        # loss_d_real_must = self.adv_loss(logits_real_must, target_is_real=True, for_discriminator=True)
        # discriminator_loss_must = (loss_d_fake_must + loss_d_real_must) * 0.5

        # logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().detach())[-1]
        # loss_d_fake_us = self.adv_loss(logits_fake_us, target_is_real=False, for_discriminator=True)
        # logits_real_us = self.discriminator_us(x_us.contiguous().detach())[-1]
        # loss_d_real_us = self.adv_loss(logits_real_us, target_is_real=True, for_discriminator=True)
        # discriminator_loss_us = (loss_d_fake_us + loss_d_real_us) * 0.5

        # discriminator_loss = discriminator_loss_must + discriminator_loss_us

        # loss_d = self.hparams.adversarial_weight * discriminator_loss

        # loss_d.backward()
        # optimizer_d.step()


        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)


        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
    
    def validation_step(self, val_batch, batch_idx):        
        real_a, real_b = val_batch

        # forward        
        fake_b, q_loss_b = self.generator_a(real_a)  # G_A(A)
        reconstruction_a, q_loss_a = self.generator_b(fake_b)   # G_B(G_A(A))
        
        fake_a, q_loss_a = self.generator_b(real_b)  # G_B(B)
        reconstruction_b, q_loss_b = self.generator_a(fake_a)   # G_A(G_B(B))

        loss_g = self.compute_loss_g(real_a, fake_b, reconstruction_a, real_b, fake_a, reconstruction_b)

        self.log("val_loss", loss_g, sync_dist=True)
        # self.log("val_loss_quantization", quantization_loss, sync_dist=True)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    

class CycleGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator_a = ResnetGenerator(input_nc=1, output_nc=1)
        self.generator_b = ResnetGenerator(input_nc=1, output_nc=1)

        self.discriminator_a = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        
        self.discriminator_b = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.cycle_loss = nn.L1Loss()
        self.idt_loss = nn.L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

    def get_b(self, x):
        return self.generator_b(x)

    def get_a(self, x):
        return self.generator_a(x)

    def configure_optimizers(self):
        g_params = list(self.generator_a.parameters()) + list(self.generator_b.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        d_params = list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters())
        optimizer_d = optim.AdamW(d_params,
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


    def compute_loss_g(self, real_a, fake_a, reconstruction_a, real_b, fake_b, reconstruction_b):
        """Calculate the loss for generators G_A and G_B"""
        loss_idt = 0
        # Identity loss
        if self.hparams.lambda_idt > 0:
            idt_a = self.generator_a(real_a)
            loss_idt_a = self.idt_loss(real_a, idt_a)
            
            idt_b = self.generator_b(self.real_b)
            loss_idt_b = self.idt_loss(real_b, idt_b)

            loss_idt = (loss_idt_a + loss_idt_b)*self.hparams.lambda_idt

        # GAN loss D_A(G_A(A))
        logits_fake_a = self.discriminator_a(fake_a.contiguous().float())[-1]
        loss_generator_a = self.adv_loss(logits_fake_a, target_is_real=True, for_discriminator=False) * self.hparams.gamma_a
        # GAN loss D_B(G_B(B))        
        logits_fake_b = self.discriminator_b(fake_b.contiguous().float())[-1]
        loss_generator_b = self.adv_loss(logits_fake_b, target_is_real=True, for_discriminator=False)  * self.hparams.gamma_b
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_a = self.cycle_loss(reconstruction_a, real_a) * self.hparams.lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_b = self.cycle_loss(reconstruction_b, real_b) * self.hparams.lambda_b
        # combined loss and calculate gradients

        perceptual_loss_a = self.perceptual_loss(fake_a.float(), real_a.float()) * self.hparams.ommega_a
        perceptual_loss_b = self.perceptual_loss(fake_b.float(), real_b.float()) * self.hparams.ommega_b

        return loss_generator_a + loss_generator_b + loss_cycle_a + loss_cycle_b + perceptual_loss_a + perceptual_loss_b + loss_idt

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
        real_a, real_b = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        # forward        
        fake_b = self.generator_b(real_a)  # G_A(A)
        reconstruction_a = self.generator_a(fake_b.clone().detach())   # G_B(G_A(A))
        
        fake_a = self.generator_a(real_b)  # G_B(B)
        reconstruction_b = self.generator_b(fake_a.clone().detach())   # G_A(G_B(B))

        # Optimize generator
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], requires_grad=False)

        optimizer_g.zero_grad()        
        loss_g = self.compute_loss_g(real_a, fake_a, reconstruction_a, real_b, fake_b, reconstruction_b)
        loss_g.backward()
        optimizer_g.step()
      
        # D_A and D_B
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], requires_grad=True)
        optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
        loss_d_a = self.compute_loss_d(self.discriminator_a, real_a, fake_a)      # calculate gradients for D_b
        loss_d_a.backward()
        loss_d_b = self.compute_loss_d(self.discriminator_b, real_b, fake_b)      # calculate gradients for D_a
        loss_d_b.backward()
        optimizer_d.step()  # update D_A and D_B's weights

        loss_d = loss_d_a + loss_d_b

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)


        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
    
    def validation_step(self, val_batch, batch_idx):        
        real_a, real_b = val_batch

        # forward        
        fake_b = self.generator_b(real_a)  # G_A(A)
        reconstruction_a = self.generator_a(fake_b)   # G_B(G_A(A))
        
        fake_a = self.generator_a(real_b)  # G_B(B)
        reconstruction_b = self.generator_b(fake_a)   # G_A(G_B(B))

        loss_g = self.compute_loss_g(real_a, fake_a, reconstruction_a, real_b, fake_b, reconstruction_b)

        self.log("val_loss", loss_g, sync_dist=True)

class CycleAutoEncoderKLV2(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl_mr_us = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.autoencoderkl_us_mr = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator_mr = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        
        self.discriminator_us = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

    def get_us(self, x_mr):
        return self.autoencoderkl_mr_us(x_mr)

    def get_mr(self, x_us):
        return self.autoencoderkl_us_mr(x_us)

    def configure_optimizers(self):
        g_params = list(self.autoencoderkl_mr_us.parameters()) + list(self.autoencoderkl_us_mr.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        d_params = list(self.discriminator_mr.parameters()) + list(self.discriminator_us.parameters())
        optimizer_d = optim.AdamW(d_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x_mr, x_us = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        optimizer_g.zero_grad()

        # # MR -> US -> MR
        # # 1. Encode the MR using the trained mr encoder
        # z_mr = self.encode_mr(x_mr)
        # # 2. Pass it through the autoencoder to transform it to the US domain
        # z_mr_us, z_mu_mr_us, z_sigma_mr_us = self.autoencoderkl_mr_us(z_mr)
        # # 3. Reconstruct an US using the trained US decoder this is the fake MR from US
        # reconstruction_mr_us = self.decode_us(z_mr_us)


        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)

        # z_us = self.encode_us(reconstruction_mr_us)
        # z_us_mr, z_mu_us_mr, z_sigma_us_mr = self.autoencoderkl_us_mr(z_us)
        # reconstruction_mr = self.decode_mr(z_us_mr)

        recons_loss_mr = F.l1_loss(reconstruction_mr.float(), x_mr.float())
        p_loss_mr = self.perceptual_loss(reconstruction_mr.float(), x_mr.float())


        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr = kl_loss_mr_us + kl_loss_us_mr


        loss_g = recons_loss_mr + (self.hparams.kl_weight * kl_loss_mr) + (self.hparams.perceptual_weight * p_loss_mr)
        

        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)


        recons_loss_us = F.l1_loss(reconstruction_us.float(), x_us.float())
        p_loss_us = self.perceptual_loss(reconstruction_us.float(), x_us.float())
        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us = kl_loss_us_mr + kl_loss_mr_us

        loss_g += recons_loss_us + (self.hparams.kl_weight * kl_loss_us) + (self.hparams.perceptual_weight * p_loss_us)


        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:


            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().float())[-1]
            generator_loss_mr = self.adversarial_loss(logits_fake_mr, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_mr

            logits_fake_mr = self.discriminator_mr(reconstruction_us_mr.contiguous().float())[-1]
            generator_loss_us_mr = self.adversarial_loss(logits_fake_mr, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_us_mr
            
            logits_fake_us = self.discriminator_us(reconstruction_mr_us.contiguous().float())[-1]
            generator_loss_mr_us = self.adversarial_loss(logits_fake_us, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_mr_us

            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().float())[-1]
            generator_loss_us = self.adversarial_loss(logits_fake_us, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_us


        loss_g.backward()
        optimizer_g.step()


        loss_d = 0.
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:            
            
            optimizer_d.zero_grad()

            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().detach())[-1]
            loss_d_fake_mr = self.adversarial_loss(logits_fake_mr, target_is_real=False, for_discriminator=True)
            logits_real_mr = self.discriminator_mr(x_mr.contiguous().detach())[-1]
            loss_d_real_mr = self.adversarial_loss(logits_real_mr, target_is_real=True, for_discriminator=True)
            logits_fake_us_mr = self.discriminator_mr(reconstruction_us_mr.contiguous().detach())[-1]
            loss_d_fake_us_mr = self.adversarial_loss(logits_fake_us_mr, target_is_real=False, for_discriminator=True)
            discriminator_loss_mr = (loss_d_fake_mr + loss_d_real_mr + loss_d_fake_us_mr) * 0.5

            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().detach())[-1]
            loss_d_fake_us = self.adversarial_loss(logits_fake_us, target_is_real=False, for_discriminator=True)
            logits_real_us = self.discriminator_us(x_us.contiguous().detach())[-1]
            loss_d_real_us = self.adversarial_loss(logits_real_us, target_is_real=True, for_discriminator=True)
            logits_fake_mr_us = self.discriminator_us(reconstruction_mr_us.contiguous().detach())[-1]
            loss_d_fake_mr_us = self.adversarial_loss(logits_fake_mr_us, target_is_real=False, for_discriminator=True)
            discriminator_loss_us = (loss_d_fake_us + loss_d_real_us + loss_d_fake_mr_us) * 0.5

            loss_d = self.hparams.adversarial_weight * (discriminator_loss_mr + discriminator_loss_us)

            loss_d.backward()
            optimizer_d.step()


        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)


        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x_mr, x_us = val_batch
        
        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)
        recon_loss_mr = F.l1_loss(x_mr.float(), reconstruction_mr.float())


        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)
        recon_loss_us = F.l1_loss(x_us.float(), reconstruction_us.float())

        recon_loss = recon_loss_mr + recon_loss_us

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.get_us(images)


class VQVAEPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vqvae = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512, 1024),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=128
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")        
        self.l1_loss = torch.nn.L1Loss()

        self.adv_weight = 0.01
        self.perceptual_weight = 0.001

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.vqvae.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, quantization_loss = self.vqvae(images=self.noise_transform(x))
        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = self.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + self.perceptual_weight * p_loss + self.adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()


        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, quantization_loss = self.vqvae(images=x)
        recon_loss = self.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images):        
        return self.vqvae(images)



class VQVAEPLFull(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vqvae = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512, 1024, 2048),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=128
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")        
        self.l1_loss = torch.nn.L1Loss()

        self.adv_weight = 0.01
        self.perceptual_weight = 0.001

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.vqvae.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, quantization_loss = self.vqvae(images=self.noise_transform(x))
        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = self.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + self.perceptual_weight * p_loss + self.adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()


        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, quantization_loss = self.vqvae(images=x)
        recon_loss = self.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images):        
        return self.vqvae(images)



class GanKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        vqvae = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            num_channels=(8, 16, 32, 64, 256, 512),
            num_res_channels=256,
            num_res_layers=1,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=self.hparams.emb_dim
        )

        self.decoder = vqvae.decoder
        self.quantizer = vqvae.quantizer

        # self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=3, out_channels=3)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.decoder.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr_d,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    

    def training_step(self, train_batch, batch_idx):
        z_mu = train_batch["z_mu"]
        z_sigma = train_batch["z_sigma"]

        z_vae = self.sampling(z_mu, z_sigma)


        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()


        fake, x_loss = self(z_vae.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss + x_loss

        loss_g.backward()
        optimizer_g.step()       

        loss_d = 0.
        
        optimizer_d.zero_grad()

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(z_vae.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}

    def validation_step(self, val_batch, batch_idx):

        z_mu = val_batch["z_mu"]
        z_sigma = val_batch["z_sigma"]

        z_vae = self.sampling(z_mu, z_sigma)

        fake, x_loss = self(z_vae.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(z_vae.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss        

        self.log("val_loss_g", loss_g, sync_dist=True)
        self.log("val_loss_d", loss_d, sync_dist=True)
        self.log("val_loss", loss_d + loss_g, sync_dist=True)


    def forward(self, num):
        x = torch.randn(num, self.hparams.emb_dim, 1, 1).to(self.device)
        x_loss, x = self.quantizer(x)
        return self.decoder(x), x_loss

        

class DDPMPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        in_channels = 3
        if hasattr(self.hparams, "in_channels"):
            in_channels = self.hparams.in_channels

        out_channels = 3
        if hasattr(self.hparams, "out_channels"):
            out_channels = self.hparams.out_channels
        
        self.model = nets.DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )

        if hasattr(self.hparams, "use_pre_trained") and self.hparams.use_pre_trained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True)


        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):

        images = train_batch

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        images = val_batch

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())


        self.log("val_loss", loss, sync_dist=True)


    def forward(self, x):

        noise = torch.randn_like(x).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps=self.hparams.num_train_timesteps)        

        return self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler, verbose=False)

class DDPMPL64(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        in_channels = 3
        if hasattr(self.hparams, "in_channels"):
            in_channels = self.hparams.in_channels

        out_channels = 3
        if hasattr(self.hparams, "out_channels"):
            out_channels = self.hparams.out_channels
        
        self.model = nets.DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )

        if hasattr(self.hparams, "use_pre_trained") and self.hparams.use_pre_trained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True)


        self.resize_transform = transforms.Resize([-1, 64, 64])
        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):

        images = train_batch
        images = self.resize_transform(images)

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        images = val_batch
        images = self.resize_transform(images)

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())


        self.log("val_loss", loss, sync_dist=True)


    def forward(self, x):

        noise = torch.randn_like(self.resize_transform(x)).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps=self.hparams.num_train_timesteps)        

        return self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler, verbose=False)


class DDPMPL64_ddim(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        in_channels = 3
        if hasattr(self.hparams, "in_channels"):
            in_channels = self.hparams.in_channels

        out_channels = 3
        if hasattr(self.hparams, "out_channels"):
            out_channels = self.hparams.out_channels
        
        self.model = nets.DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )

        if hasattr(self.hparams, "use_pre_trained") and self.hparams.use_pre_trained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True)


        self.resize_transform = transforms.Resize([-1, 64, 64])
        self.scheduler = DDIMScheduler(num_train_timesteps=self.hparams.num_train_timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):

        images = train_batch
        images = self.resize_transform(images)

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        images = val_batch
        images = self.resize_transform(images)

        noise = torch.randn_like(images)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Get model prediction
        noise_pred = self.inferer(inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())


        self.log("val_loss", loss, sync_dist=True)


    def forward(self, x):

        noise = torch.randn_like(self.resize_transform(x)).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps=self.hparams.num_train_timesteps)        

        return self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler, verbose=False)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)  
        x = torch.abs(x)      
        return F.normalize(x, dim=1)

class Diffusion_AE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.unet = nets.DiffusionModelUNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    num_channels=(128, 256, 256),
                    attention_levels=(False, True, True),
                    num_res_blocks=1,
                    num_head_channels=64,
                    with_conditioning=True,
                    cross_attention_dim=1,
                )
        
        if hasattr(self.hparams, 'base_encoder'):
            # template_model = getattr(torchvision.models, self.hparams.base_encoder)
            # self.semantic_encoder = template_model(num_classes=4*self.hparams.emb_dim)

            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
            model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
            self.semantic_encoder = template_model(**model_params)
            
            if hasattr(self.semantic_encoder, 'classifier'):
                self.semantic_encoder.classifier = nn.Sequential(
                    self.semantic_encoder.classifier,
                    ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
                )            

            elif hasattr(self.semantic_encoder, 'fc'):
            
                self.semantic_encoder.fc = nn.Sequential(
                    self.semantic_encoder.fc,  # Linear(ResNet output, 4*hidden_dim)
                    ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
                )
        else:
            self.semantic_encoder = torchvision.models.resnet18()
            self.semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.semantic_encoder.fc = torch.nn.Linear(512, self.hparams.emb_dim)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        
        self.inferer = DiffusionInferer(self.scheduler)

    # def forward(self, xt, x_cond, t=100):
    #     latent = self.semantic_encoder(x_cond)
    #     noise_pred = self.unet(x=xt, timesteps=t, context=latent.unsqueeze(2))
    #     return noise_pred, latent

    def forward(self, x, timesteps=100):
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(num_inference_steps=timesteps)        
        noise = torch.randn_like(x).to(self.device)
        latent = self.semantic_encoder(x)
        return self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=scheduler, save_intermediates=False, conditioning=latent.unsqueeze(2), verbose=False)        
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):

        images = train_batch

        batch_size = images.shape[0]
        
        noise = torch.randn_like(images).to(self.device)
        # Create timesteps
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batch_size,)).to(self.device).long()
        # Get model prediction
        # cross attention expects shape [batch size, sequence length, channels], we are use channels = latent dimension and sequence length = 1
        latent = self.semantic_encoder(images)
        noise_pred = self.inferer(inputs=images, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        images = val_batch

        batch_size = images.shape[0]

        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batch_size,)).to(self.device).long()

        noise = torch.randn_like(images).to(self.device)
        latent = self.semantic_encoder(images)
        noise_pred = self.inferer(inputs=images, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("val_loss", loss, sync_dist=True)
        
    

class Diffusion_AE_EncoderKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.unet = nets.DiffusionModelUNet(
                    spatial_dims=2,
                    in_channels=3,
                    out_channels=3,
                    num_channels=(128, 256, 256),
                    attention_levels=(False, True, True),
                    num_res_blocks=1,
                    num_head_channels=64,
                    with_conditioning=True,
                    cross_attention_dim=1,
                )
        self.semantic_encoder = torchvision.models.resnet18()
        self.semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.semantic_encoder.fc = torch.nn.Linear(512, self.hparams.emb_dim)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        
        self.inferer = DiffusionInferer(self.scheduler)
        
        
        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=self.hparams.latent_channels,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )

        self.resize_transform = torchvision.transforms.Resize(64)

    def forward(self, x, timesteps=100):
        z_mu, z_sigma = self.autoencoderkl.encode(x)
        # z_mu = z_mu.detach()
        # z_sigma = z_sigma.detach()

        scheduler = DDIMScheduler()
        scheduler.set_timesteps(num_inference_steps=timesteps)        
        noise = torch.randn_like(z_mu).to(self.device)
        latent = self.semantic_encoder(self.resize_transform(x))
        return self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=scheduler, save_intermediates=False, conditioning=latent.unsqueeze(2), verbose=False)        
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):

        images = train_batch

        batch_size = images.shape[0]

        with torch.no_grad():
            z_mu, z_sigma = self.autoencoderkl.encode(images)
            z_mu = z_mu.detach()
            # z_sigma = z_sigma.detach()
            # z_vae = self.autoencoderkl.sampling(z_mu, z_sigma)
        
        images = self.resize_transform(images)
        
        noise = torch.randn_like(z_mu).to(self.device)
        # Create timesteps
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batch_size,)).to(self.device).long()
        # Get model prediction
        # cross attention expects shape [batch size, sequence length, channels], we are use channels = latent dimension and sequence length = 1
        latent = self.semantic_encoder(images)

        noise_pred = self.inferer(inputs=z_mu, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        images = val_batch

        batch_size = images.shape[0]

        z_mu, z_sigma = self.autoencoderkl.encode(images)
        # z_vae = self.autoencoderkl.sampling(z_mu, z_sigma)

        images = self.resize_transform(images)
        
        noise = torch.randn_like(z_mu).to(self.device)
        # Create timesteps
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batch_size,)).to(self.device).long()
        # Get model prediction
        # cross attention expects shape [batch size, sequence length, channels], we are use channels = latent dimension and sequence length = 1
        latent = self.semantic_encoder(images)

        noise_pred = self.inferer(inputs=z_mu, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("val_loss", loss, sync_dist=True)
    