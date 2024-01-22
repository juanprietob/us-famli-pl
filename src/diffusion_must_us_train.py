import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.mr_dataset import MRDatasetVolumes
from loaders.ultrasound_dataset import USDataset
from loaders.mr_us_dataset import MUSTUSDataModule
from transforms.ultrasound_transforms import MustEvalTransforms, MustTrainTransforms, RealUSTrainTransforms, RealEvalTransforms
from callbacks.logger import ImageLoggerMustUSNeptune

from nets import diffusion

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

import warnings
warnings.filterwarnings("ignore", message="Divide by zero (a_min == a_max)")

def main(args):

    if(os.path.splitext(args.csv_train_must)[1] == ".csv"):
        df_train_must = pd.read_csv(args.csv_train_must)
        df_val_must = pd.read_csv(args.csv_valid_must)   
    else:
        df_train_must = pd.read_parquet(args.csv_train_must)
        df_val_must = pd.read_parquet(args.csv_valid_must)   

    if(os.path.splitext(args.csv_train_us)[1] == ".csv"):
        df_train_us = pd.read_csv(args.csv_train_us)
        df_val_us = pd.read_csv(args.csv_valid_us)   
    else:
        df_train_us = pd.read_parquet(args.csv_train_us)
        df_val_us = pd.read_parquet(args.csv_valid_us)   

    NN = getattr(diffusion, args.nn)    
    model = NN(**vars(args))


    train_transform_must = MustTrainTransforms(height=args.height)
    valid_transform_must = MustEvalTransforms(height=args.height)

    must_ds_train = USDataset(df_train_must, mount_point=args.mount_point, img_column='img_path', transform=train_transform_must, repeat_channel=False)
    must_ds_val = USDataset(df_val_must, mount_point=args.mount_point, img_column='img_path', transform=valid_transform_must, repeat_channel=False)

    train_transform_us = RealUSTrainTransforms(height=args.height)
    valid_transform_us = RealEvalTransforms(height=args.height)

    us_ds_train = USDataset(df_train_us, args.mount_point, img_column='img_path', transform=train_transform_us, repeat_channel=False)
    us_ds_val = USDataset(df_val_us, args.mount_point, img_column='img_path', transform=valid_transform_us, repeat_channel=False)

    must_us_data = MUSTUSDataModule(must_ds_train, must_ds_val, us_ds_train, us_ds_val, batch_size=args.batch_size, num_workers=args.num_workers)


    # must_us_data.setup()

    # train_ds = must_us_data.train_dataloader()
    # for idx, batch in enumerate(train_ds):
    #     must, us = batch

    #     print("__")
    #     print(must.shape)
    #     print(us.shape)
    #     print("..")


    # quit()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    logger = None

    if args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/DiffusionMustUS',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = ImageLoggerMustUSNeptune(log_steps=args.log_steps)
        callbacks.append(image_logger)

    trainer = Trainer(
        logger=logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        # plugins=[MixedPrecisionPlugin(precision='16-mixed', device='cuda')],
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    trainer.fit(model, datamodule=must_us_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=2)
    # hparams_group.add_argument('--random_slice_size', help='Number of random slice to grab from volumes, affects batch size, i.e., batch_size=random_slice_size*batch_size', type=int, default=10)          
    hparams_group.add_argument('--ommega_a', help='Perceptual weight a', type=float, default=0.001)
    hparams_group.add_argument('--ommega_b', help='Perceptual weight b', type=float, default=0.001)
    hparams_group.add_argument('--gamma_a', help='Generator weight a', type=float, default=1.0)
    hparams_group.add_argument('--gamma_b', help='Generator weight b', type=float, default=1.0)
    hparams_group.add_argument('--lambda_a', help='Lambda a reconstruction weight', type=float, default=10.0)
    hparams_group.add_argument('--lambda_b', help='Lambda b reconstruction weight', type=float, default=10.0)
    hparams_group.add_argument('--lambda_idt', help='Lambda b reconstruction weight', type=float, default=0.0)
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--kl_weight', help='Weight decay for optimizer', type=float, default=1e-6)    
    hparams_group.add_argument('--autoencoder_warm_up_n_epochs', help='Warmup epochs', type=float, default=10)



    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="CycleAutoEncoderKL")
    input_group.add_argument('--nn_mr', help='Type of neural network MR', type=str, default="AutoEncoderKL")
    input_group.add_argument('--nn_us', help='Type of neural network US', type=str, default="AutoEncoderKL")
    # input_group.add_argument('--model_mr', help='Trained MR autoencoder model', type=str, required=True)
    # input_group.add_argument('--model_us', help='Trained US autoencoder model', type=str, required=True)
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train_must', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid_must', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_train_us', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid_us', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--height', type=int, default=256, help='Image size')    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)


    args = parser.parse_args()

    main(args)
