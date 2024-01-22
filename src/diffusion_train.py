import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import USDataModule, SimuDataModule, USDataModuleV2
from transforms.ultrasound_transforms import DiffusionV2EvalTransforms, DiffusionV2TrainTransforms, DiffusionEvalTransforms, DiffusionTrainTransforms, DiffusionTrainTransformsPaired, DiffusionEvalTransformsPaired
from callbacks.logger import DiffusionImageLogger, DiffusionImageLoggerNeptune, DiffusionImageLoggerPairedNeptune

from nets import diffusion

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_test))
    else:
        df_train = pd.read_parquet(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_parquet(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv_test))


    NN = getattr(diffusion, args.nn)
    model = NN(**vars(args))


    # train_transform = DiffusionTrainTransforms()
    # valid_transform = DiffusionEvalTransforms()

    # if args.target_column:
        
    #     train_transform = DiffusionTrainTransformsPaired()
    #     valid_transform = DiffusionEvalTransformsPaired()

    #     usdata = SimuDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform, valid_transform=valid_transform, repeat_channel=False, target_column=args.target_column)
    # else:

    #     usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform, valid_transform=valid_transform, repeat_channel=False)

    train_transform = DiffusionV2TrainTransforms()
    valid_transform = DiffusionV2EvalTransforms()

    usdata = USDataModuleV2(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform, valid_transform=valid_transform)

    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=args.save_top_k,
        save_last=True,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks = [early_stop_callback, checkpoint_callback]
    logger = None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)  
        image_logger = DiffusionImageLogger(num_images=args.num_images)  
        callbacks.append(image_logger)

    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/DiffusionUS',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        if args.target_column:
            image_logger = DiffusionImageLoggerPairedNeptune(num_images=args.num_images)
        else:
            image_logger = DiffusionImageLoggerNeptune(num_images=args.num_images)
        callbacks.append(image_logger)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        # plugins=[MixedPrecisionPlugin(precision='16-mixed', device='cuda')],
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--perceptual_weight', help='Perceptual weight', type=float, default=0.001)
    hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=0.01)
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--kl_weight', help='Weight decay for optimizer', type=float, default=1e-6)    
    hparams_group.add_argument('--autoencoder_warm_up_n_epochs', help='Warmup epochs', type=float, default=10)        
    hparams_group.add_argument('--num_train_timesteps', help='Num train steps for ddpml', type=int, default=1000)
    hparams_group.add_argument('--latent_channels', help='Latent Channels', type=int, default=3)    
    hparams_group.add_argument('--denoise', help='Use noise transform during training', type=int, default=1)    
    hparams_group.add_argument('--smooth', help='Use smooth transform during training', type=int, default=1)    

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="AutoEncoderKL")
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--img_column', help='Name of the column with source images', type=str, default="img_path")    
    input_group.add_argument('--target_column', help='Name of the column for the target images', type=str, default=None)    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--save_top_k', help='Save k top models', type=int, default=2)
    
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--num_images', help='Max number of images', type=int, default=16)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")


    args = parser.parse_args()

    main(args)
