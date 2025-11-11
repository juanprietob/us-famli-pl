import argparse
import os

from loaders import ultrasound_dataset

from nets import efw
from lightning import Trainer

from lightning.pytorch.loggers import NeptuneLogger

def main(args):

    NN = getattr(efw, args.nn)    
    model = NN.load_from_checkpoint(args.model)

    DM = getattr(ultrasound_dataset, model.hparams.data_module)    

    model.hparams.csv_test = args.csv
    datamodule = DM(**model.hparams)

    logger_neptune = None
    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/fetal-biometry',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='EFW Test', add_help=False)

    input_group = parser.add_argument_group('Input')

    input_group.add_argument('--csv', help='CSV file path', type=str, default='/mnt/raid/C1_ML_Analysis/CSV_files/efw_2025-10-31_test.csv')
    input_group.add_argument('--nn', help='Type of neural network', type=str, required=True)
    input_group.add_argument('--model', help='Model for testing', type=str, default=None)

    input_group.add_argument('--neptune_tags', help='Neptune tags for logging', type=str, nargs='+', default=None)
    
    args = parser.parse_args()

    main(args)
