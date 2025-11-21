import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from .layers import TimeDistributed, AttentionChunk, MHABlock, SelfAttention, ProjectionHead

import torchvision 
from torchvision.transforms import v2


from lightning.pytorch import LightningModule

from lightning.pytorch.loggers import NeptuneLogger
from neptune.types import File

import torchmetrics
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics.aggregation import CatMetric
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

import os
import json
import math

import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues, experiment=None, out_path=None):
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    fig_cm = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()    

    if experiment:
        experiment.upload(fig_cm)
    if out_path:
        plt.savefig(out_path)
    plt.close(fig_cm)

def compute_classification_report(targets, probs, experiment=None, out_path=None):
    report_dict = classification_report(targets, probs, digits=3, output_dict=True)
    report_txt = classification_report(targets, probs, digits=3)
    print(report_txt)

    if experiment:
        experiment.upload(
            File.from_content(report_txt, extension="txt")
        )        
    if(out_path):
        with open(out_path, "+w") as f:
            json.dump(report_dict, f, indent=4)

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EfficientNet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        NN = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = NN(num_classes=self.hparams.num_classes)

        self.extract_features = False

        if hasattr(self.hparams, 'model_feat') and self.hparams.model_feat is not None:
            classifier = self.convnet.classifier
            self.convnet.classifier = nn.Identity()
            self.convnet.load_state_dict(torch.load(args.model_feat))
            # for param in self.convnet.parameters():
            #     param.requires_grad = False
            self.convnet.classifier = classifier


        class_weights = None
        if(hasattr(self.hparams, 'class_weights')):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(std=0.05)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = self(self.noise_transform(x))

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("test_acc", self.accuracy)

    def forward(self, x):        
        if self.extract_features:
            x_f = self.convnet.features(x)
            x_f = self.convnet.avgpool(x_f)            
            x = torch.flatten(x_f, 1)            
            return self.convnet.classifier(x), x_f
        else:
            return self.convnet(x)


class EffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Identity()
        self.proj_final = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.num_classes, activation=nn.LeakyReLU)

        weights = None if self.hparams.class_weights is None else torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.class_names = self.hparams.class_names if hasattr(self.hparams, 'class_names') and self.hparams.class_names else range(self.hparams.num_classes)

        self.probs = CatMetric()
        self.targets = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Simple Classification Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')

        group.add_argument("--num_classes", type=int, default=3, help='Number of output classes for the model')        
        group.add_argument("--class_names", type=str, default=None, help='Class names')        
        group.add_argument("--class_weights", nargs="+", default=None, type=float, help='Class weights for the loss function')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X_hat, Y)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        batch_size = Y.size(0)
        self.accuracy(X_hat, Y)
        self.log(f"{step}_acc", self.accuracy, batch_size=batch_size, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y = train_batch["class"]

        x_hat = self(X)

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        Y = val_batch["class"]

        x_hat = self(X) 

        self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)
        self.conf.update(x_hat, Y)

    
    def on_validation_epoch_end(self):
        
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y = test_batch["class"]

        x_hat = self(X)

        self.conf.update(x_hat, Y)

        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)

    def on_test_epoch_end(self):

        confmat  = self.conf.compute()
        probs = self.probs.compute()
        targets = self.targets.compute()

        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

    def forward(self, x: torch.tensor):

        z = self.model(x)
        return self.proj_final(z)
    
class EffnetV2sSoft(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Identity()
        self.proj_final = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.num_classes, activation=nn.LeakyReLU)

        weights = None if self.hparams.class_weights is None else torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.class_names = self.hparams.class_names if hasattr(self.hparams, 'class_names') and self.hparams.class_names else range(self.hparams.num_classes)

        self.probs = CatMetric()
        self.targets = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Simple Classification Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')

        group.add_argument("--num_classes", type=int, default=3, help='Number of output classes for the model')        
        group.add_argument("--class_names", type=str, default=None, help='Class names')        
        group.add_argument("--class_weights", nargs="+", default=None, type=float, help='Class weights for the loss function')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X_hat, Y)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        batch_size = Y.size(0)
        self.accuracy(X_hat, Y)
        self.log(f"{step}_acc", self.accuracy, batch_size=batch_size, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y = train_batch["scalar"]

        x_hat = self(X)

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        Y = val_batch["scalar"]

        x_hat = self(X) 

        self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)
        self.conf.update(x_hat, Y.argmax(dim=-1))

    
    def on_validation_epoch_end(self):
        
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")
                    
            compute_classification_report(targets.argmax(dim=-1).cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y = test_batch["scalar"]

        x_hat = self(X)

        self.conf.update(x_hat, Y.argmax(dim=-1))

        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)

    def on_test_epoch_end(self):

        confmat  = self.conf.compute()
        probs = self.probs.compute()
        targets = self.targets.compute()

        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.argmax(dim=-1).cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

    def forward(self, x: torch.tensor):

        z = self.model(x)
        return self.proj_final(z)

class RegEffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Identity()

        self.encoder = TimeDistributed(encoder)

        self.proj = nn.Linear(self.hparams.features, 1)
        self.act = nn.Sigmoid()

        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
                v2.RandomResizedCrop(size=256, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )

        self.loss_fn = nn.L1Loss(reduction='sum')

        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()

        self.preds = CatMetric()
        self.targets = CatMetric()


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):

        # print("Y unique:", torch.unique(Y))
        # print("Y min/max:", Y.min().item(), Y.max().item())
        # print("Number of zeros in Y:", (Y == 0).sum().item())

        # print("X_hat min/max:", X_hat.min().item(), X_hat.max().item())
        # print("Number of zeros in X_hat:", (X_hat == 0).sum().item())
        
        loss = (self.loss_fn(X_hat, Y.float())*((Y*8.0)*(Y>=0.75) + 1.0)).sum()
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        self.mae(X_hat, Y)
        self.mse(X_hat, Y)

        self.log(f"{step}_mae", self.mae, sync_dist=sync_dist)
        self.log(f"{step}_mse", self.mse, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)

        x_hat = self(self.train_transform(X))

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        Y = val_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)

        x_hat = self(X)

        self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)

        self.preds.update(x_hat.view(-1))
        self.targets.update(Y.view(-1))

    def on_validation_epoch_end(self):
        
        preds = self.preds.compute().view(-1)
        targets = self.targets.compute().view(-1)

        # compute metrics
        mae = self.mae.compute()
        mse = self.mse.compute()
        # reset metrics for next epoch
        self.mae.reset()
        self.mse.reset()

        # OPTIONAL: create a scatter plot y_true vs y_pred and upload to Neptune
        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            fig, ax = plt.subplots()
            ax.scatter(targets.cpu(), preds.cpu(), alpha=0.4)
            ax.plot([0, 1.1], [0, 1.1])  # identity line
            ax.set_xlabel("True quality score")
            ax.set_ylabel("Predicted quality score")
            ax.set_title("Validation: y_true vs y_pred")

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_regression_scatter"]
                experiment.upload(fig)
            plt.close(fig)

            # Example JSON report
            report_dict = {
                "mae": float(mae.cpu()),
                "mse": float(mse.cpu()),
                "rmse": float(torch.sqrt(mse).cpu())
            }

            if isinstance(logger, NeptuneLogger):
                run["reports/val_regression_report"].upload(
                    File.from_content(json.dumps(report_dict), extension="json")
                )
            else:
                print(json.dumps(report_dict, indent=4))

        self.preds.reset()
        self.targets.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)

        x_hat = self(X)
        
        self.preds.update(x_hat.view(-1))
        self.targets.update(Y.view(-1))

    def on_test_epoch_end(self):

        preds = self.preds.compute().view(-1)
        targets = self.targets.compute().view(-1)

        # compute metrics
        mae = self.mae.compute()
        mse = self.mse.compute()
        # reset metrics for next epoch
        self.mae.reset()
        self.mse.reset()

        # OPTIONAL: create a scatter plot y_true vs y_pred and upload to Neptune
        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            fig, ax = plt.subplots()
            ax.scatter(targets.cpu(), preds.cpu(), alpha=0.4)
            ax.plot([0, 1.1], [0, 1.1])  # identity line
            ax.set_xlabel("True quality score")
            ax.set_ylabel("Predicted quality score")
            ax.set_title("Validation: y_true vs y_pred")

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_regression_scatter"]
                experiment.upload(fig)
            plt.close(fig)

            # Example JSON report
            report_dict = {
                "mae": float(mae.cpu()),
                "mse": float(mse.cpu()),
                "rmse": float(torch.sqrt(mse).cpu())
            }

            if isinstance(logger, NeptuneLogger):
                run["reports/test_regression_report"].upload(
                    File.from_content(json.dumps(report_dict), extension="json")
                )
            else:
                print(json.dumps(report_dict, indent=4))

        self.preds.reset()
        self.targets.reset()

    def forward(self, x: torch.tensor):

        z = self.encoder(x)
        return self.act(self.proj(z).squeeze(-1))


class FlyToClassification(LightningModule):
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

        self.ln0 = nn.LayerNorm(self.hparams.embed_dim)
        self.mha = MHABlock(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, causal_mask=False, return_weights=False)
        self.ln1 = nn.LayerNorm(self.hparams.embed_dim)

        self.dropout = nn.Dropout(self.hparams.dropout)
        
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=self.hparams.num_classes, activation=nn.PReLU)

        weights = None if self.hparams.class_weights is None else torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()

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
        group = parent_parser.add_argument_group("Fetal Anatomy time aware classification Model")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-5)
        
        # Image Encoder parameters                 
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--time_dim_train", type=int, nargs="+", default=(16, 128), help='Range of time dimensions for training')
        group.add_argument("--n_chunks_e", type=int, default=2, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=16, help='Number of outputs in the time dimension, this will determine the first dimension of the 2D positional encoding')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')
        
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')        
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--tags", type=int, default=18, help='Number of sweep tags for the sequences, this will determine the second dimension of the 2D positional encoding')

        group.add_argument("--num_classes", type=int, default=3, help='Number of output classes for the model')        
        group.add_argument("--class_weights", nargs="+", default=None, type=float, help='Class weights for the loss function')
        

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
        H = -(s.clamp(eps,1-eps)*torch.log(s.clamp(eps,1-eps)) + (1-s.clamp(eps,1-eps))*torch.log(1-s.clamp(eps,1-eps)))
        return H.mean()

    def regularizer(self, scores, lam_l1=1e-3, lam_ent=1e-4):
        return lam_l1 * scores.mean() + lam_ent * self.entropy_penalty(scores)

    # def regularizer(self, scores, lam_l1=1e-3, lam_bi=1e-3):
    #     return lam_l1 * scores.mean() + lam_bi * (scores * (1 - scores)).mean()
    # def regularizer(self, scores, lam_l1=1e-3, lam_bi=1e-3):
        
    #     return 0.0
    
    def compute_loss(self, Y, X_hat, X_s=None, step="train", sync_dist=False):

        batch_size, T, Cl = X_hat.shape

        Y = Y.view(-1)
        X_hat = X_hat.view(-1, Cl)

        loss = self.loss_fn(X_hat, Y) 
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        
        self.accuracy(X_hat, Y)
        self.log(f"{step}_acc", self.accuracy, batch_size=batch_size, sync_dist=sync_dist)

        if X_s is not None:
            X_s = X_s.view(-1)
            self.log(f"{step}_scores/mean", X_s.mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/max", X_s.max(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.9", (X_s >= 0.9).float().mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.5", (X_s >= 0.5).float().mean(), sync_dist=sync_dist)

            reg_loss = self.regularizer(X_s)
            # Y_s = (Y > 0).float()
            # reg_loss = ((X_s - Y_s)**2).mean()

            self.log(f"{step}_loss_reg", reg_loss, sync_dist=sync_dist)
            loss = loss + reg_loss

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        tags = train_batch["tag"]
        Y = train_batch["class"]
        
        batch_size, C, T, H, W = X.shape
        time_r = torch.randint(low=self.hparams.time_dim_train[0], high=self.hparams.time_dim_train[1], size=(1,)).item()
        time_ridx = torch.randint(low=0, high=T, size=(time_r,))
        time_ridx = time_ridx.sort().values
        X = X[:, :, time_ridx, :, :].contiguous()
        Y = Y[:, time_ridx].contiguous()

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat, z_t_s = self(self.train_transform(X), tags)

        return self.compute_loss(Y=Y, X_hat=x_hat, X_s=z_t_s, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        tags = val_batch["tag"]
        Y = val_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat, z_t_s = self(X, tags) 

        self.compute_loss(Y=Y, X_hat=x_hat, X_s=z_t_s, step="val", sync_dist=True)

        Y = Y.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)
        self.conf.update(x_hat, Y)

    
    def on_validation_epoch_end(self):
        
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            fig_cm = plt.figure()
            plt.imshow(confmat.cpu().numpy(), interpolation='nearest')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.tight_layout()

            if isinstance(self.trainer.logger, NeptuneLogger):
                self.trainer.logger.experiment["images/val_Confusion_Matrix"].upload(fig_cm)
            plt.close(fig_cm)

            print(classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), digits=3))

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        tags = test_batch["tag"]
        Y = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]            

        x_hat, _ = self(X, tags)

        Y = Y.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)

        self.conf.update(x_hat, Y)

        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)

    def on_test_epoch_end(self):

        confmat  = self.conf.compute()
        probs = self.probs.compute()
        targets = self.targets.compute()

        if self.trainer.is_global_zero:

            fig_cm = plt.figure()
            plt.imshow(confmat.cpu().numpy(), interpolation='nearest')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.tight_layout()

            if isinstance(self.trainer.logger, NeptuneLogger):
                self.trainer.logger.experiment["images/Confusion_Matrix"].upload(fig_cm)
            plt.close(fig_cm)

            print(classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), digits=3))

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

        # x_sweeps shape is B, T, C, H, W. N for number of sweeps ex. torch.Size([2, 200, 3, 256, 256])
        # tags shape torch.Size([2, 2])

        z = self.encode(x_sweeps) # [BS, T, self.hparams.features]

        z = self.proj(z) # [BS, T, self.hparams.embed_dim]
        
        z_t, z_t_s = self.attn_chunk(z) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

        p_enc_z = self.p_encoding_z[sweeps_tags]            
        
        z_t = z_t + p_enc_z
        z_t = z_t + self.mha(self.ln0(z_t)) #[BS, self.hparams.n_chunks, self.hparams.embed_dim]
        z_t = self.ln1(z_t)

        z_t, z_s = self.attn(z_t, z_t)
        
        z = z + self.dropout(z_t.unsqueeze(1))  # [BS, T, self.hparams.embed_dim]
        
        x_hat = self.proj_final(z)

        return x_hat, z_t_s