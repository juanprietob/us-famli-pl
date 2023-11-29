import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy

from pl_bolts.models.self_supervised import Moco_v2


class USMoco(Moco_v2):
    def init_encoders(self, base_encoder):


        template_model = getattr(torchvision.models, base_encoder)
        encoder_q = template_model(num_classes=self.hparams.emb_dim)
        encoder_k = template_model(num_classes=self.hparams.emb_dim)

        if hasattr(encoder_q, 'classifier'):

            dim_mlp = encoder_q.classifier[1].weight.shape[1]
            encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder_q.classifier[1])
            encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder_k.classifier[1])

        return encoder_q, encoder_k

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(trainer.strategy, DDPStrategy)

class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr=1e-3, temperature=0.1, weight_decay=1e-4, max_epochs=500, base_encoder='resnet18'):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)

        template_model = getattr(torchvision.models, base_encoder)
        self.convnet = template_model(num_classes=4*hidden_dim)

        # if hasattr(self.convnet, 'classifier'):
        #     self.convnet.classifier = nn.Sequential(
        #         self.convnet.classifier,
        #         nn.ReLU(inplace=True),
        #         nn.Linear(4*hidden_dim, hidden_dim)
        #     )            

        # elif hasattr(self.convnet, 'fc'):
        
        #     self.convnet.fc = nn.Sequential(
        #         self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
        #         nn.ReLU(inplace=True),
        #         nn.Linear(4*hidden_dim, hidden_dim)
        #     )

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.hidden_dim, hidden_dim=4*self.hparams.hidden_dim, output_dim=self.hparams.hidden_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.hidden_dim, hidden_dim=4*self.hparams.hidden_dim, output_dim=self.hparams.hidden_dim)
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')

    def forward(self, x):
        x = self.convnet(x)
        return x

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

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x

class Sim(pl.LightningModule):

    def __init__(self, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, k=128, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

    def on_train_epoch_start(self):        
        self.noise_transform[0].std = self.hparams.beta*1.0/torch.exp(torch.tensor(self.hparams.alpha*self.current_epoch))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def compute_loss(self, x_0, x_1, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_proj = torch.sum(torch.square(1.0 - self.loss(z_0, z_1)))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        if mode == 'val':
            k = batch_size
        else:
            k = min(batch_size, self.hparams.k)        

        loss_proj_c = self.loss(z_0_r, z_1)
        top_k_i = torch.topk(1.0 - loss_proj_c, k=k).indices
        loss_proj_c = torch.sum(torch.square(loss_proj_c[top_k_i]))
        
        loss =  loss_proj + loss_proj_c
        
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        x_0, x_1 = batch
        return self.compute_loss(x_0, x_1, mode='train')

    def validation_step(self, batch, batch_idx):
        x_0, x_1 = batch
        self.compute_loss(x_0, x_1, mode='val')

    def forward(self, x):
        return self.convnet(x)

class SimScore(pl.LightningModule):

    def __init__(self, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, k=128, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50, drop_last_dim=False, hidden_dim=None):
        super().__init__()
        self.save_hyperparameters()

        if (self.hparams.hidden_dim == None):
            self.hparams.hidden_dim = 4*self.hparams.emb_dim

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

        self.n = self.north(emb_dim)



    def north(self, emb_dim):
        north = torch.zeros(1, emb_dim)
        north[:, -1] = 1
        return north  


    def on_fit_start(self):        
        self.n  = self.n.to(self.device)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def compute_loss(self, x_0, x_1, score, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_proj = torch.sum(torch.square(1.0 - self.loss(z_0, z_1)))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        if mode == 'val':
            k = batch_size
        else:
            k = min(batch_size, self.hparams.k)        

        if self.hparams.drop_last_dim:
            loss_proj_c = self.loss(z_0_r[:,0:-1], z_1[0,0:-1])
        else:
            loss_proj_c = self.loss(z_0_r, z_1)
        top_k_i = torch.topk(1.0 - loss_proj_c, k=k).indices
        loss_proj_c = torch.sum(torch.square(loss_proj_c[top_k_i]))


        # Compute the cosine similarity between the z_0/z_1 and the north pole
        # We want bad frames to map towards the north pole
        # 1.0 - self.loss(z_0, north) this will be close to 0 if z0 maps close to the north pole
        # If the score is 1 or closer to 1, we want those frames to map far from the north pole
        # score - (1.0 - self.loss(z_0, north))
        loss_north = torch.sum(torch.square(score - (1.0 - self.loss(z_0, self.n)) + score - (1.0 - self.loss(z_1, self.n))))
        
        loss =  loss_proj + loss_proj_c + loss_north
        
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss_north', loss_north)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.compute_loss(x_0, x_1, score, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.compute_loss(x_0, x_1, score, mode='val')

    def forward(self, x):
        return self.convnet(x)

class SimScoreW(pl.LightningModule):

    def __init__(self, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, w=4.0, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

        self.n = self.north(emb_dim)

    def north(self, emb_dim):
        north = torch.zeros(1, emb_dim)
        north[:, -1] = 1
        return north  


    def on_fit_start(self):        
        self.n  = self.n.to(self.device)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
        # return optimizer

    def compute_loss(self, x_0, x_1, score, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_proj = torch.sum(torch.square(1.0 - self.loss(z_0, z_1)))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
        loss_proj_c = self.loss(z_0_r, z_1)
        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first

        w = torch.square(torch.arange(batch_size, device=self.device)/batch_size - 1.0)*self.hparams.w

        loss_proj_c = torch.sum(w*torch.square(loss_proj_c[loss_proj_c_sorted_i]))


        # Compute the cosine similarity between the z_0/z_1 and the north pole
        # We want bad frames to map towards the north pole
        # 1.0 - self.loss(z_0, north) this will be close to 0 if z0 maps close to the north pole
        # If the score is 1 or closer to 1, we want those frames to map far from the north pole
        # score - (1.0 - self.loss(z_0, north))
        loss_north = torch.sum(torch.square(score - (1.0 - self.loss(z_0, self.n)) + score - (1.0 - self.loss(z_1, self.n))))
        
        loss =  loss_proj + loss_proj_c + loss_north
        
        self.log(mode + '_loss_proj', loss_proj, sync_dist=True)
        self.log(mode + '_loss_proj_c', loss_proj_c, sync_dist=True)
        self.log(mode + '_loss_north', loss_north, sync_dist=True)
        self.log(mode + '_loss', loss, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.compute_loss(x_0, x_1, score, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.compute_loss(x_0, x_1, score, mode='val')

    def forward(self, x):
        return self.convnet(x)

class SimScoreWK(pl.LightningModule):

    def __init__(self, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, w=4.0, k=224, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=4*self.hparams.emb_dim, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

        self.n = self.north(emb_dim)

    def north(self, emb_dim):
        north = torch.zeros(1, emb_dim)
        north[:, -1] = 1
        return north  


    def on_fit_start(self):        
        self.n  = self.n.to(self.device)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
        # return optimizer

    def compute_loss(self, x_0, x_1, score, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_proj = torch.sum(torch.square(1.0 - self.loss(z_0, z_1)))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
        loss_proj_c = self.loss(z_0_r, z_1)
        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first

        w = torch.square(torch.arange(batch_size, device=self.device)/batch_size - 1.0)*self.hparams.w        

        loss_proj_c = w*torch.square(loss_proj_c[loss_proj_c_sorted_i])

        if mode == 'val':
            k = batch_size
        else:
            k = min(batch_size, self.hparams.k)

        loss_proj_c = torch.sum(loss_proj_c[0:k])


        # Compute the cosine similarity between the z_0/z_1 and the north pole
        # We want bad frames to map towards the north pole
        # 1.0 - self.loss(z_0, north) this will be close to 0 if z0 maps close to the north pole
        # If the score is 1 or closer to 1, we want those frames to map far from the north pole
        # score - (1.0 - self.loss(z_0, north))
        loss_north = torch.sum(torch.square(score - (1.0 - self.loss(z_0, self.n)) + score - (1.0 - self.loss(z_1, self.n))))
        
        loss =  loss_proj + loss_proj_c + loss_north
        
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss_north', loss_north)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.compute_loss(x_0, x_1, score, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.compute_loss(x_0, x_1, score, mode='val')

    def forward(self, x):
        return self.convnet(x)


class SimScoreOnlyW(pl.LightningModule):
    def __init__(self, args=None, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, w=4.0, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=64, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=64, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
        # return optimizer

    def compute_loss(self, x_0, x_1, score, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)


        loss_proj = self.loss(z_0, z_1)

        loss_proj_mean = torch.mean(loss_proj)
        loss_proj_std = torch.std(loss_proj)
        
        loss_proj = torch.sum(torch.square(1.0 - loss_proj))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
        loss_proj_c = self.loss(z_0_r, z_1)


        loss_proj_c_mean = torch.mean(loss_proj_c)
        loss_proj_c_std = torch.std(loss_proj_c)

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first
        loss_proj_c = loss_proj_c[loss_proj_c_sorted_i] 
        
        w = torch.pow(torch.arange(batch_size, device=self.device)/batch_size, 2)*self.hparams.w

        loss_proj_c = torch.sum(w*torch.square(loss_proj_c))
        
        
        loss =  loss_proj + loss_proj_c
        
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_mean', loss_proj_mean)
        self.log(mode + '_loss_proj_std', loss_proj_std)        
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss_proj_c_mean', loss_proj_c_mean)
        self.log(mode + '_loss_proj_c_std', loss_proj_c_std)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.compute_loss(x_0, x_1, score, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.compute_loss(x_0, x_1, score, mode='val')

    def forward(self, x):
        return self.convnet(x)


class SimScoreOnlyWExp(pl.LightningModule):
    def __init__(self, args=None, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, w0=4.0, w1=-10, weight_decay=1e-4, max_epochs=50, hidden_dim=64):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def compute_loss(self, x_0, x_1, score, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)


        loss_proj = self.loss(z_0, z_1)

        loss_proj_mean = torch.mean(loss_proj)
        loss_proj_std = torch.std(loss_proj)
        
        loss_proj = torch.sum(torch.square(1.0 - loss_proj))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
        loss_proj_c = self.loss(z_0_r, z_1)


        loss_proj_c_mean = torch.mean(loss_proj_c)
        loss_proj_c_std = torch.std(loss_proj_c)

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first
        loss_proj_c = loss_proj_c[loss_proj_c_sorted_i] 
        
        # w = torch.pow(torch.arange(batch_size, device=self.device)/batch_size, 2)*self.hparams.w
        # NOTE: W1 must be negative
        w = self.hparams.w0*torch.exp(self.hparams.w1*torch.arange(batch_size, device=self.device)/batch_size)

        loss_proj_c = torch.sum(w*torch.square(loss_proj_c))
        
        
        loss =  loss_proj + loss_proj_c
        
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_mean', loss_proj_mean)
        self.log(mode + '_loss_proj_std', loss_proj_std)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss_proj_c_mean', loss_proj_c_mean)
        self.log(mode + '_loss_proj_c_std', loss_proj_c_std)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.compute_loss(x_0, x_1, score, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.compute_loss(x_0, x_1, score, mode='val')

    def forward(self, x):
        return self.convnet(x)

class SimNorth(pl.LightningModule):
    def __init__(self, args=None, light_house=None):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.args.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.args.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.args.emb_dim, hidden_dim=self.hparams.args.hidden_dim, output_dim=self.hparams.args.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.args.emb_dim, hidden_dim=self.hparams.args.hidden_dim, output_dim=self.hparams.args.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )
        
        if light_house is None:
            light_house = torch.rand(self.hparams.args.n_lights, self.hparams.args.emb_dim)
        else:
            light_house = torch.tensor(light_house)
        # print(light_house)
        # light_house = torch.normal(mean=0.5, std=0.1)

        self.register_buffer('light_house', light_house)

        min_l = torch.tensor(999999999)
        for idx, l in enumerate(light_house):
            lights_ex = torch.cat([light_house[:idx], light_house[idx+1:]])
            min_l = torch.minimum(min_l, torch.min(torch.sum(torch.square(l - lights_ex), dim=1)))

        self.noise_transform_lights = torch.nn.Sequential(
            GaussianNoise(mean=0.0, std=min_l.item()/2.0)
        )


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.args.lr,
                                weight_decay=self.hparams.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.args.epochs, eta_min=self.hparams.args.lr/50)
        return [optimizer], [lr_scheduler]

    # def compute_loss(self, x_0, x_1, score, mode):
    #     batch_size = x_0.size(0)
        
    #     x = torch.cat([x_0, x_1], dim=0)

    #     z = self(self.noise_transform(x))

    #     z_0, z_1 = torch.split(z, batch_size)


    #     loss_proj = self.loss(z_0, z_1)

    #     loss_proj_mean = torch.mean(loss_proj)
    #     loss_proj_std = torch.std(loss_proj)
        
    #     loss_proj = torch.sum(torch.square(1.0 - loss_proj))

    #     #randomize the batch        
    #     r = torch.randperm(batch_size)
    #     z_0_r = z_0[r]
    #     # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
    #     # 1. Get the most different ones. topk(1 - loss)
    #     # 2. We want to further maximize their distance
    #     # 2.1 We get the indices of those elements
    #     # 2.2 We sum those elements, we want that their cosine similarit is  0
    #     # If all images in the batch are mapping to the same direction, this term should be really high.
    #     # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
    #     loss_proj_c = self.loss(z_0_r, z_1)


    #     loss_proj_c_mean = torch.mean(loss_proj_c)
    #     loss_proj_c_std = torch.std(loss_proj_c)

    #     loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first
    #     loss_proj_c = loss_proj_c[loss_proj_c_sorted_i] 

    #     # similar_imgs_idx = torch.where(loss_proj_c > (loss_proj_mean + self.hparams.w_dev*loss_proj_std))[0]
    #     # if similar_imgs_idx.nelement() > 0:
    #     #     delta = (batch_size - similar_imgs_idx[0])/batch_size
    #     # else:
    #     #     delta = 1.0

    #     # delta = self.hparams.delta

    #     # w = torch.pow(delta - torch.arange(batch_size, device=self.device)/batch_size, 3)*self.hparams.w
    #     # w = torch.pow(torch.arange(batch_size, device=self.device)/batch_size, 2)*self.hparams.w
    #     # NOTE: W1 must be negative
    #     w = self.hparams.args.w0*torch.exp(self.hparams.args.w1*torch.arange(batch_size, device=self.device)/batch_size)
    #     loss_proj_c = torch.sum(w*torch.square(loss_proj_c))


    #     n0 = self.clusters_mean[torch.randint(low=0, high=self.hparams.args.n_clusters, size=(1,))]
    #     n1 = self.clusters_mean[torch.randint(low=0, high=self.hparams.args.n_clusters, size=(1,))]


    #     loss_north0 = self.loss(z_0, n0)
    #     loss_north0_mean = torch.mean(loss_north0)
    #     loss_north0_std = torch.std(loss_north0)        
    #     loss_north0_sorted_i = torch.argsort(loss_north0)# this gives the indices sorted in ascending order, i.e., the most different ones first
    #     loss_north0 = loss_north0[loss_north0_sorted_i] 

    #     loss_north1 = self.loss(z_1, n1)
    #     loss_north1_mean = torch.mean(loss_north1)
    #     loss_north1_std = torch.std(loss_north1)
    #     loss_north1_sorted_i = torch.argsort(loss_north1)
    #     loss_north1 = loss_north1[loss_north1_sorted_i] 

    #     w = self.hparams.args.w2*torch.pow(self.hparams.args.w3 - torch.arange(batch_size, device=self.device)/batch_size,3)
    #     loss_north = torch.sum(w*torch.square(loss_north0)) + torch.sum(w*torch.square(loss_north1))        
        
    #     loss =  loss_proj + loss_proj_c + loss_north
        
    #     self.log(mode + '_loss_proj', loss_proj)
    #     self.log(mode + '_loss_proj_mean', loss_proj_mean)
    #     self.log(mode + '_loss_proj_std', loss_proj_std)        
    #     self.log(mode + '_loss_proj_c', loss_proj_c)
    #     self.log(mode + '_loss_proj_c_mean', loss_proj_c_mean)
    #     self.log(mode + '_loss_proj_c_std', loss_proj_c_std)
    #     self.log(mode + '_loss', loss)
        
    #     self.log(mode + '_loss_north0_mean', loss_north0_mean)
    #     self.log(mode + '_loss_north0_std', loss_north0_std)
        
    #     self.log(mode + '_loss_north1_mean', loss_north1_mean)
    #     self.log(mode + '_loss_north1_std', loss_north1_std)

    #     self.log(mode + '_loss_north', loss_north)
        
    #     return loss

    def compute_loss(self, x_0, x_1, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)


        loss_proj = self.loss(z_0, z_1)

        loss_proj_mean = torch.mean(loss_proj)
        loss_proj_std = torch.std(loss_proj)
        
        loss_proj = torch.sum(torch.square(1.0 - loss_proj))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        
        loss_proj_c = self.loss(z_0_r, z_1)


        loss_proj_c_mean = torch.mean(loss_proj_c)
        loss_proj_c_std = torch.std(loss_proj_c)

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most different one first
        loss_proj_c = loss_proj_c[loss_proj_c_sorted_i] 
           
        w = torch.square(torch.arange(batch_size, device=self.device)/batch_size - 1.0)*self.hparams.args.w        
        # w = self.hparams.args.w0*torch.exp(self.hparams.args.w1*torch.arange(batch_size, device=self.device)/batch_size)
        loss_proj_c = torch.sum(w*torch.square(loss_proj_c))

        loss_north = []

        light_house = self.noise_transform_lights(self.light_house)
        
        for lh in light_house:            
            l_north = self.loss(z_0, lh)
            l_north_sorted_i_z0 = torch.argsort(l_north)# this gives the indices sorted in ascending order, i.e., the most different ones first            
            l_north_sorted_i_z0 = l_north_sorted_i_z0[-1] # get the most similar one            

            z_0_n = l_north[l_north_sorted_i_z0]
            loss_north.append(1. - z_0_n)#we want to bring them closer together

            l_north = self.loss(z_1, lh)
            l_north_sorted_i_z1 = torch.argsort(l_north)# this gives the indices sorted in ascending order, i.e., the most different ones first
            l_north_sorted_i_z1 = l_north_sorted_i_z1[-1] # get the most similar one
            z_1_n = l_north[l_north_sorted_i_z0]
            loss_north.append(1. - z_1_n) #we want to bring them closer together

        loss_north = torch.stack(loss_north)
        loss_north_mean = torch.mean(loss_north)
        loss_north_std = torch.std(loss_north)
        loss_north = torch.square(torch.sum(loss_north))

        loss =  loss_proj + loss_proj_c + loss_north
        
        self.log(mode + '_loss_proj', loss_proj, sync_dist=True)
        self.log(mode + '_loss_proj_mean', loss_proj_mean, sync_dist=True)
        self.log(mode + '_loss_proj_std', loss_proj_std, sync_dist=True)
        self.log(mode + '_loss_proj_c', loss_proj_c, sync_dist=True)
        self.log(mode + '_loss_proj_c_mean', loss_proj_c_mean, sync_dist=True)
        self.log(mode + '_loss_proj_c_std', loss_proj_c_std, sync_dist=True)
        self.log(mode + '_loss', loss, sync_dist=True)
        
        self.log(mode + '_loss_north_mean', loss_north_mean, sync_dist=True)
        self.log(mode + '_loss_north_std', loss_north_std, sync_dist=True)
        self.log(mode + '_loss_north', loss_north, sync_dist=True)

        # self.log(mode + '_loss_north_c_mean', loss_north_c_mean)
        # self.log(mode + '_loss_north_c_std', loss_north_c_std)
        # self.log(mode + '_loss_north_c', loss_north_c)
        
        return loss

    def training_step(self, batch, batch_idx):
        x_0, x_1 = batch
        return self.compute_loss(x_0, x_1, mode='train')

    def validation_step(self, batch, batch_idx):
        x_0, x_1 = batch
        self.compute_loss(x_0, x_1, mode='val')

    def forward(self, x):
        return self.convnet(x)


#Changes to loss function according to Wang(2020)
class ModSimScoreOnlyW(pl.LightningModule):
    def __init__(self, args=None, base_encoder='efficientnet_b0', emb_dim=128, lr=1e-3, w=4.0, alpha=0.1, beta=0.3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        self.lam = 1.0
        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4*self.hparams.emb_dim)

        if hasattr(self.convnet, 'classifier'):
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=64, output_dim=self.hparams.emb_dim)
            )            

        elif hasattr(self.convnet, 'fc'):
        
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                ProjectionHead(input_dim=4*self.hparams.emb_dim, hidden_dim=64, output_dim=self.hparams.emb_dim)
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def lalign(self, x, y, alpha=2):
        return (x - y).norm(dim=1).pow(alpha).mean()
    
    def lunif(self, x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def alignUnifLoss(self, x, y, mode):

        alignUnifLoss = self.lalign(x, y) + self.lam * (self.lunif(x) + self.lunif(y)) / 2
        self.log(mode + '_loss_proj', alignUnifLoss)

    def training_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        return self.alignUnifLoss(x_0, x_1, mode='train')

    def validation_step(self, batch, batch_idx):
        (x_0, x_1), score = batch
        self.alignUnifLoss(x_0, x_1, mode='val')

    def forward(self, x):
        return self.convnet(x)