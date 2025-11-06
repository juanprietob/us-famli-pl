import argparse
import os
import torch

from loaders import ultrasound_dataset
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import classification
from callbacks import logger

import lightning as L

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger

def main(args):

    deterministic = None
    if args.seed_everything:
        seed_everything(args.seed_everything, workers=True)
        deterministic = True

    NN = getattr(classification, args.nn)    
    model = NN(**vars(args))

    DM = getattr(ultrasound_dataset, args.data_module)    

    datamodule = DM(**vars(args))

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True
        
    )

    callbacks.append(checkpoint_callback)

    if args.monitor:
        checkpoint_callback_d = ModelCheckpoint(
            dirpath=args.out,
            filename='{epoch}-{' + args.monitor + ':.2f}',
            save_top_k=5,
            monitor=args.monitor,
            save_last=True
            
        )

        callbacks.append(checkpoint_callback_d)


    if args.use_early_stopping:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
        callbacks.append(early_stop_callback)

    logger_neptune = None

    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/us-cl',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )

    if args.logger:
        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(log_steps=args.image_log_steps)
        callbacks.append(image_logger)
    
    trainer = Trainer(
        logger=logger_neptune,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=args.find_unused_parameters),
        deterministic=deterministic
        # strategy=DDPStrategy(),
    )
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.model)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification training', add_help=False)

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--seed_everything', help='Seed everything for training', type=int, default=None)
    hparams_group.add_argument('--find_unused_parameters', help='find_unused_parameters', type=int, default=0)

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, required=True)        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)    

    input_group.add_argument('--data_module', help='Type of data module to use', type=str, required=True)            
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=1)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, default=None)
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=5)
    log_group.add_argument('--image_log_steps', help='Log images every N steps', type=int, default=50)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(classification, args.nn)    
    NN.add_model_specific_args(parser)

    data_module = getattr(ultrasound_dataset, args.data_module)
    parser = data_module.add_data_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_args()

    main(args)
