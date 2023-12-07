import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.mr_dataset import MRDatasetVolumes
from loaders.ultrasound_dataset import USDataset
from loaders.mr_us_dataset import VolumeSlicingProbeParamsDataset, MUSTUSDataModule
from transforms.ultrasound_transforms import LabelTrainTransforms, LabelEvalTransforms, RealUSTrainTransforms, RealEvalTransforms
from callbacks.logger import ImageLoggerLotusNeptune

from nets import lotus

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

import SimpleITK as sitk


def main(args):

    if(os.path.splitext(args.csv_train_params)[1] == ".csv"):
        df_train_params = pd.read_csv(args.csv_train_params)
        df_val_params = pd.read_csv(args.csv_valid_params)   
    else:
        df_train_params = pd.read_parquet(args.csv_train_label)
        df_val_params = pd.read_parquet(args.csv_valid_label)   

    if(os.path.splitext(args.csv_train_us)[1] == ".csv"):
        df_train_us = pd.read_csv(args.csv_train_us)
        df_val_us = pd.read_csv(args.csv_valid_us)   
    else:
        df_train_us = pd.read_parquet(args.csv_train_us)
        df_val_us = pd.read_parquet(args.csv_valid_us)   

    NN = getattr(lotus, args.nn)    
    model = NN(**vars(args))


    train_transform_label = LabelTrainTransforms()
    valid_transform_label = LabelEvalTransforms()

    img_label = sitk.ReadImage(args.labeled_img)

    labeled_ds_train = VolumeSlicingProbeParamsDataset(volume=img_label, df=df_train_params, mount_point=args.mount_point, transform=train_transform_label)
    labeled_ds_val = VolumeSlicingProbeParamsDataset(volume=img_label, df=df_val_params, mount_point=args.mount_point, transform=valid_transform_label)

    train_transform_us = RealUSTrainTransforms()
    valid_transform_us = RealEvalTransforms()

    us_ds_train = USDataset(df_train_us, args.mount_point, img_column='img_path', transform=train_transform_us, repeat_channel=False)
    us_ds_val = USDataset(df_val_us, args.mount_point, img_column='img_path', transform=valid_transform_us, repeat_channel=False)

    must_us_data = MUSTUSDataModule(labeled_ds_train, labeled_ds_val, us_ds_train, us_ds_val, batch_size=args.batch_size, num_workers=args.num_workers)


    must_us_data.setup()

    # train_ds = must_us_data.train_dataloader()
    # for idx, batch in enumerate(train_ds):
    #     label, us = batch

    #     print("__")
    #     print(label.shape)
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
            project='ImageMindAnalytics/Lotus',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = ImageLoggerLotusNeptune(log_steps=args.log_steps)
        callbacks.append(image_logger)

    trainer = Trainer(
        logger=logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
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
    hparams_group.add_argument('--num_labels', help='Number of labels in the US model', type=int, default=340)
    hparams_group.add_argument('--alpha_coeff_boundary_map', help='Lotus model', type=float, default=0.1)
    hparams_group.add_argument('--beta_coeff_scattering', help='Lotus model', type=float, default=0.1)
    hparams_group.add_argument('--tgc', help='Lotus model', type=int, default=8)
    hparams_group.add_argument('--clamp_vals', help='Lotus model', type=int, default=0)
    hparams_group.add_argument('--parceptual_weight', help='Perceptual weight', type=float, default=1.0)
    hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=1.0)    
    hparams_group.add_argument('--warm_up_n_epochs', help='Number of warm up epochs before starting to train with discriminator', type=int, default=5)
    
    
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--kl_weight', help='Weight decay for optimizer', type=float, default=1e-6)    


    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="RealUS")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train_params', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid_params', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--labeled_img', required=True, type=str, help='Labeled volume to grap slices from')    
    input_group.add_argument('--csv_train_us', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid_us', required=True, type=str, help='Valid CSV')    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)


    args = parser.parse_args()

    main(args)
