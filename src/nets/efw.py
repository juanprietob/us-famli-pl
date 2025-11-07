import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from .layers import TimeDistributed, AttentionChunk, MHABlock, SelfAttention, ProjectionHead, HeteroscedasticHead

import torchvision 
from torchvision.transforms import v2


from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm


from lightning.pytorch.loggers import NeptuneLogger
from neptune.types import File


import matplotlib.pyplot as plt

import os
import json
import math

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

def gaussian_nll(y, mu, log_var, reduction="mean"):
    """
    Gaussian Negative Log-Likelihood for heteroscedastic regression.
    NLL = 0.5 * ( (y-mu)^2 / var + log_var )    (constant log(2π) omitted)
    """
    var = torch.exp(log_var)
    nll = 0.5 * ((y - mu)**2 / var + log_var)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


class EfwNet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Identity()
        self.encoder = TimeDistributed(encoder)
        
        p_encoding_z = torch.stack([self.positional_encoding(self.hparams.n_chunks, self.hparams.embed_dim, tag) for tag in range(self.hparams.tags)])
        self.register_buffer("p_encoding_z", p_encoding_z)
        
        self.proj = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.embed_dim, activation=nn.PReLU)
        self.attn_chunk = AttentionChunk(input_dim=self.hparams.embed_dim, hidden_dim=64, chunks=self.hparams.n_chunks)

        self.ln = nn.LayerNorm(self.hparams.embed_dim)
        self.mha = MHABlock(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, causal_mask=True, return_weights=False)

        self.dropout = nn.Dropout(self.hparams.dropout)
        
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=1, activation=nn.PReLU)
        # self.proj_final = HeteroscedasticHead(input_dim=self.hparams.embed_dim, hidden=64)

        self.loss_fn = nn.HuberLoss(delta=0.7)
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = gaussian_nll
        self.l1_fn = torch.nn.L1Loss()
        

        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),                
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(size=256)]),
                    v2.RandomResizedCrop(size=256, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                ]),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Fetal EFW time aware Model")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-5)
        
        # Image Encoder parameters                 
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--time_dim_train", type=int, nargs="+", default=(64, 96), help='Range of time dimensions for training')
        group.add_argument("--n_chunks_e", type=int, default=2, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=16, help='Number of outputs in the time dimension, this will determine the first dimension of the 2D positional encoding')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')
        
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')        
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--tags", type=int, default=18, help='Number of sweep tags for the sequences, this will determine the second dimension of the 2D positional encoding')
        group.add_argument("--loss_reg_weight", type=float, default=1.0, help='Weight for regularization loss')
        group.add_argument("--rho", type=float, default=0.1, help='Target sparsity for the regularization. Mean of scores should approach rho')
        group.add_argument("--lam_ms", type=float, default=1.0, help='Weight for mean sparsity controls the mean of the scores')
        group.add_argument("--lam_ent", type=float, default=1.0, help='Weight for entropy regularization loss, controls pushing scores toward 0 or 1')

        group.add_argument("--output_dim", type=int, default=1, help='Output dimension')

        return parent_parser
    
    def positional_encoding(self, seq_len: int, d_model: int, tag: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            tag (int): Unique tag for the sequence.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(tag * seq_len, (tag + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def entropy_penalty(self, s, eps=1e-8):
        s = s.clamp(eps, 1 - eps)
        H = -(s * torch.log(s) + (1 - s) * torch.log(1 - s))
        return H.mean() 
    
    def kl_to_bernoulli_mean(self, scores, rho=0.05, eps=1e-8):
        # KL( Bern(mean(scores)) || Bern(rho) )
        m = scores.clamp(eps, 1 - eps).mean()
        rho_t = torch.tensor(rho, device=self.device)
        kl = m * torch.log(m / rho_t) + (1 - m) * torch.log((1 - m) / (1 - rho_t))
        return kl  

    def regularizer(self, scores, rho=0.05, lam_ms=1.0, lam_ent=1.0):
        # KL controls the mean, entropy pushes toward {0,1}
        # reg = lam_kl * self.kl_to_bernoulli_mean(scores, rho=rho)
        topk = torch.topk(scores, k=int(rho * scores.numel())).values        
        bottomk = torch.topk(-scores, k=int((1 - rho) * scores.numel())).values

        reg = lam_ms * ((1.0 - topk.mean()) ** 2 + (bottomk.mean()) ** 2)
        reg += lam_ent * self.entropy_penalty(scores)
        return reg

    def compute_loss(self, Y, X_hat, X_s=None, step="train", sync_dist=False):

        loss = self.loss_fn(Y, X_hat)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        
        l1 = self.l1_fn(X_hat, Y)
        self.log(f"{step}_l1", l1, sync_dist=sync_dist)

        if X_s is not None:

            X_s = X_s.view(-1)
            self.log(f"{step}_scores/mean", X_s.mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/max", X_s.max(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.9", (X_s >= 0.9).float().mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.5", (X_s >= 0.5).float().mean(), sync_dist=sync_dist)            

            reg_loss = self.regularizer(X_s, rho=self.hparams.rho, lam_ms=self.hparams.lam_ms, lam_ent=self.hparams.lam_ent)*self.hparams.loss_reg_weight

            self.log(f"{step}_loss_reg", reg_loss, sync_dist=sync_dist)
            loss = loss + reg_loss

        return loss
    
    def on_before_optimizer_step(self, optimizer):        
        norms = grad_norm(self.mha, norm_type=2)
        self.log_dict(norms)

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        tags = train_batch["tag"]
        Y = train_batch["efw"]
        
        batch_size, NS, C, T, H, W = X.shape
        time_r = torch.randint(low=self.hparams.time_dim_train[0], high=self.hparams.time_dim_train[1], size=(1,)).item()
        time_ridx = torch.randint(low=0, high=T, size=(time_r,))
        time_ridx = time_ridx.sort().values
        X = X[:, :, :, time_ridx, :, :].contiguous()

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, z_t_s = self(self.train_transform(X), tags)

        return self.compute_loss(Y=Y, X_hat=x_hat, X_s=z_t_s, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        tags = val_batch["tag"]
        Y = val_batch["efw"]

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, z_t_s = self(X, tags)

        self.compute_loss(Y=Y, X_hat=x_hat, X_s=z_t_s, step="val", sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        tags = test_batch["tag"]
        Y = test_batch["efw"]

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, z_t_s = self(X, tags)

        self.compute_loss(Y=Y, X_hat=x_hat, X_s=z_t_s, step="test", sync_dist=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        # z = []
        # for x_chunk in x.tensor_split(self.hparams.n_chunks_e, dim=1):            
        #     z.append(self.encoder(x_chunk))
        # z = torch.cat(z, dim=1)

        return self.encoder(x)

    def forward(self, x_sweeps: torch.tensor, sweeps_tags: torch.tensor):
        
        batch_size = x_sweeps.shape[0]

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 200, 3, 256, 256]) 
        # tags shape torch.Size([2, 2])
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T

        z = []
        z_t = []
        z_t_s = []

        for n in range(Nsweeps):

            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, T, C, H, W]
            
            tag = sweeps_tags[:,n]    

            z_ = self.encode(x_sweeps_n) # [BS, T, self.hparams.features]

            z_ = self.proj(z_) # [BS, T, self.hparams.embed_dim]

            z.append(z_)
            
            z_t_, z_t_s_ = self.attn_chunk(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

            p_enc_z = self.p_encoding_z[tag]
            
            z_t_ = z_t_ + p_enc_z
            z_t_ = self.dropout(z_t_)

            z_t_ = self.mha(self.ln(z_t_)) #[BS, self.hparams.n_chunks, self.hparams.embed_dim]

            z_t.append(z_t_)
            z_t_s.append(z_t_s_)


        z = torch.stack(z, dim=1)  # [BS, N_sweeps, T, self.hparams.embed_dim]
        z_t = torch.stack(z_t, dim=1)  # [BS, N_sweeps, self.hparams.n_chunks, self.hparams.embed_dim]
        z_t_s = torch.stack(z_t_s, dim=1)  # [BS, N_sweeps, T, self.hparams.n_chunks]

        z_t = z_t.view(batch_size, -1, self.hparams.embed_dim)  # [BS, N_s*n_chunks, self.hparams.embed_dim]
        z_t_s = z_t_s.view(batch_size, -1)  # [BS, N_s*n_chunks]

        z_t, z_s = self.attn(z_t, z_t)
        
        # x_hat_mu, x_hat_log_var = self.proj_final(z_t)
        x_hat = self.proj_final(z_t)

        return x_hat, z_t_s