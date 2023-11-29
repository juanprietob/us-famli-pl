import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import USDataModule, USDataModuleBlindSweep
from transforms.ultrasound_transforms import Moco2TrainTransforms, Moco2EvalTransforms, SimCLRTrainTransforms, SimCLREvalTransforms, EffnetDecodeTrainTransforms, EffnetDecodeEvalTransforms, SimTrainTransforms, SimEvalTransforms, SimTrainTransformsV2, SimTrainTransformsV3
from callbacks.logger import MocoImageLogger, SimCLRImageLogger, EffnetDecodeImageLogger, SimImageLogger, SimScoreImageLogger, SimNorthImageLogger
from nets.contrastive import USMoco, SimCLR, Sim, SimScore, SimScoreW, SimScoreWK, SimScoreOnlyW, SimScoreOnlyWExp, SimNorth, ModSimScoreOnlyW
from nets.hyper_sphere import LightHouse


from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

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

    if args.query:
        query = args.query
        df_train = df_train.query(query).reset_index(drop=True)
        df_val = df_val.query(query).reset_index(drop=True)
        df_test = df_test.query(query).reset_index(drop=True)
    
    if args.nn == "moco_v2":

        # train_transform = Moco2TrainTransforms(224)
        # valid_transform = Moco2EvalTransforms(224)

        train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)

        model = USMoco(base_encoder=args.base_encoder, emb_dim=args.emb_dim, softmax_temperature=args.temperature, learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, batch_size=args.batch_size, use_mlp=args.use_mlp, encoder_momentum=args.encoder_momentum)
        image_logger = MocoImageLogger()

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', ga_column='ga_boe', train_transform=train_transform, valid_transform=valid_transform, test_transform=valid_transform, drop_last=True)    


    elif args.nn == "simclr":
        model = SimCLR(hidden_dim=args.emb_dim, lr=args.lr, temperature=args.temperature, weight_decay=args.weight_decay, max_epochs=args.epochs, base_encoder=args.base_encoder)
        image_logger = SimCLRImageLogger()

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', ga_column='ga_boe', train_transform=SimTrainTransforms(224), valid_transform=SimEvalTransforms(224), drop_last=True)

    elif args.nn == "sim":
        train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = Sim(base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, k=args.k, weight_decay=args.weight_decay, max_epochs=args.epochs, drop_last_dim=args.drop_last_dim)
        image_logger = SimImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform)
    elif args.nn == "simscore":
        train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = SimScore(base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, k=args.k, weight_decay=args.weight_decay, max_epochs=args.epochs, drop_last_dim=args.drop_last_dim)
        image_logger = SimScoreImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')    
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')

    elif args.nn == "simscorew":
        train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = SimScoreW(base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, w=args.w, weight_decay=args.weight_decay, max_epochs=args.epochs)
        image_logger = SimScoreImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')    
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')
    elif args.nn == "simscorewk":
        train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = SimScoreWK(base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, w=args.w, k=args.k, weight_decay=args.weight_decay, max_epochs=args.epochs)
        image_logger = SimScoreImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')    
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')

    elif args.nn == "simscoreonlyw":
        if(args.train_transform == 2):
            print("USING TRAIN TRANSFORM 2")
            train_transform = SimTrainTransformsV2(224)    
        else:
            train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = SimScoreOnlyW(args, base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, w=args.w, delta=args.delta, weight_decay=args.weight_decay, max_epochs=args.epochs)
        image_logger = SimScoreImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')    
    
    #based on paper Wang(2020)
    elif args.nn == "modsimscoreonlyw":
        # if(args.train_transform == 2):
        #     print("USING TRAIN TRANSFORM 2")
        #     train_transform = SimTrainTransformsV2(224)    
        # else:
        #     train_transform = SimTrainTransformsV3(224)
        # valid_transform = SimEvalTransforms(224)        

        # model = ModSimScoreOnlyW(args, base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, w=args.w, delta=args.delta, weight_decay=args.weight_decay, max_epochs=args.epochs)
        # image_logger = SimScoreImageLogger()
        

        # usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')    
        model = ModSimScoreOnlyW(hidden_dim=args.emb_dim, lr=args.lr, temperature=args.temperature, weight_decay=args.weight_decay, max_epochs=args.epochs, base_encoder=args.base_encoder)
        image_logger = SimCLRImageLogger()

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', ga_column='ga_boe', train_transform=SimTrainTransformsV3(224), valid_transform=SimEvalTransforms(224), drop_last=True)


    elif args.nn == "simscoreonlywexp":

        if(args.train_transform == 2):
            print("USING TRAIN TRANSFORM 2")
            train_transform = SimTrainTransformsV2(224)    
        else:
            train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)        

        model = SimScoreOnlyWExp(args, base_encoder=args.base_encoder, emb_dim=args.emb_dim, lr=args.lr, w0=args.w0, w1=args.w1, weight_decay=args.weight_decay, max_epochs=args.epochs, hidden_dim=args.hidden_dim)
        image_logger = SimScoreImageLogger()
        

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, scalar_column='score')

    elif args.nn == "simnorth":

        if(args.train_transform == 2):
            print("USING TRAIN TRANSFORM 2")
            train_transform = SimTrainTransformsV2(224)    
        else:
            train_transform = SimTrainTransforms(224)
        valid_transform = SimEvalTransforms(224)


        lights = None
        if args.lights is not None:
            lights = pickle.load(open(args.lights, 'rb'))

        model = SimNorth(args, light_house=lights)
        image_logger = SimNorthImageLogger()

        usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/LiteAUS',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Contrastive learning training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    parser.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="moco_v2")
    parser.add_argument('--train_transform', help='Transforms', type=int, default=0)
    parser.add_argument('--emb_dim', help='Embedding dimension', type=int, default=128)
    parser.add_argument('--hidden_dim', help='Hidden dimension for the projection head', type=int, default=None)
    parser.add_argument('--drop_last_dim', help='Drop last dimension for contrastive simscore', type=bool, default=False)
    parser.add_argument('--encoder_momentum', help='Encoder Momentum', type=float, default=0.999)
    parser.add_argument('--momentum', help='Momentum for optimizer', type=float, default=0.9)
    parser.add_argument('--weight_decay', help='Weight decay for optimization', type=float, default=1e-4)
    parser.add_argument('--temperature', help='Temperature, used for the models softmax_temperature in moco_v2', type=float, default=0.07)
    parser.add_argument('--use_mlp', help='MLP Head', type=bool, default=False)    
    parser.add_argument('--k', help='Top k values for sim approach. How many samples do you expect to be similar? k should be batch_size - similar samples in batch', type=int, default=128)
    parser.add_argument('--w', help='W value to calculate the weight coefficients for the contrastive loss part w*(x/batch_size - 1)^2 x in [0, batch_size]', type=float, default=4.0)
    parser.add_argument('--w0', help='W0 weight for the exp weight function w0*e^-w1x', type=float, default=4.0)
    parser.add_argument('--w1', help='W1 weight for the exp weight function w0*e^-w1x', type=float, default=-10.0)
    parser.add_argument('--w2', help='W2 for push pull', type=float, default=10.0)
    parser.add_argument('--w3', help='W3 for push pull', type=float, default=0.5)
    parser.add_argument('--delta', help='Step to move cluster mean', type=float, default=1e-4)
    parser.add_argument('--n_lights', help='Number of light houses', type=int, default=64)
    parser.add_argument('--lights', help='Pickle file with lights', type=str, default=None)
    # parser.add_argument('--w_dev', help='mean_similarity + w_dev*std_similarity. Contrast values above that threshold will be pulled together instead of appart', type=float, default=1.0)    
    parser.add_argument('--base_encoder', help='What encoder to use', type=str, default='efficientnet_b0')
    parser.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="moco_v2")
    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    parser.add_argument('--query', default=None, type=str, help='Query to filter the dataframes')


    args = parser.parse_args()

    main(args)
