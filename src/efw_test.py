import argparse
import os
import torch

from loaders import ultrasound_dataset
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import efw
from callbacks import logger

import lightning as L

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger

def main(args):

    NN = getattr(efw, args.nn)    
    model = NN.load_from_checkpoint(args.model)

    DM = getattr(ultrasound_dataset, model.hparams.data_module)    

    model.hparams.csv_test = '/mnt/raid/C1_ML_Analysis/CSV_files/efw_2025-10-31_test.csv'
    model.hparams.num_frames_test = 128
    datamodule = DM(**model.hparams)

    logger_neptune = None
    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/fetal-biometry',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
    
    trainer = Trainer(
        logger=logger_neptune,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
    )
    
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='EFW Test', add_help=False)

    input_group = parser.add_argument_group('Input')

    input_group.add_argument('--nn', help='Type of neural network', type=str, required=True)
    input_group.add_argument('--model', help='Model for testing', type=str, default=None)

    input_group.add_argument('--neptune_tags', help='Neptune tags for logging', type=str, nargs='+', default=None)
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    args = parser.parse_args()

    main(args)
