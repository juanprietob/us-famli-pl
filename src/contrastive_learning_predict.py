import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn 
from loaders.ultrasound_dataset import USDataset
from transforms.ultrasound_transforms import Moco2TestTransforms, SimCLRTestTransforms, SimTestTransforms, USEvalSimTransforms, USEvalRealTransforms
# from pl_bolts.models.self_supervised import Moco_v2
from nets.contrastive import USMoco, SimCLR, Sim, SimScore, SimNorth
from tqdm import tqdm

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

    transform = None
    if args.nn == "moco_v2":
        model = USMoco(base_encoder=args.base_encoder, emb_dim=args.emb_dim).load_from_checkpoint(args.model)
        model = model.encoder_q.eval()
        transform = SimTestTransforms(224)

    elif args.nn == "simclr":
        model = SimCLR(hidden_dim=args.emb_dim, base_encoder=args.base_encoder).load_from_checkpoint(args.model)
        
        transform = SimCLRTestTransforms(224)

    elif args.nn == "sim":
        model = Sim(emb_dim=args.emb_dim, base_encoder=args.base_encoder).load_from_checkpoint(args.model)
        
        transform = SimTestTransforms(224)
    elif args.nn == "simscore":
        model = SimScore.load_from_checkpoint(args.model)
        if args.transform == "sim":
            transform = USEvalSimTransforms(224, scale_a_max=args.scale_a_max)
        else:
            transform = USEvalRealTransforms(224)
    elif args.nn == "simnorth":
        model = SimNorth(args).load_from_checkpoint(args.model)
        
        transform = SimTestTransforms(224)

    # if args.compute_feat:
    #     if hasattr(model, 'classifier'):
    #         model.classifier = nn.Identity()
    #     elif hasattr(model, 'fc'):
    #         model.fc = nn.Identity()
    
    model.eval()
    model.cuda() 
    
    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))


    if args.query:
        df_test = df_test.query(args.query)

    test_ds = USDataset(df_test, transform=transform, img_column=args.img_column)    

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    with torch.no_grad():

        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):             
            X = X.cuda(non_blocking=True).contiguous()           
            feat = model(X)
            features.append(feat.cpu())

        features = torch.cat(features, dim=0)
        features = features.numpy()

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    ext = os.path.splitext(args.csv)[1]
        
    out_name = os.path.join(out_dir, os.path.basename(args.csv)).replace(ext, ".pickle")

    if args.query:
        out_name = out_name.replace('.pickle', "_" + args.query + '.pickle')

        out_csv = out_name.replace(".pickle", ".csv")
        df_test.to_csv(out_csv, index=False)

    # if args.compute_feat:
    #     out_name = out_name.replace(".pickle", "_features.pickle")

    pickle.dump(features, open(out_name, 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')    
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)    
    parser.add_argument('--query', type=str, help='Apply a query to the dataframe', default=None)
    parser.add_argument('--img_column', type=str, help='CSV file for testing', default="img_path")    
    parser.add_argument('--model', help='Trained model checkpoint', type=str, required=True)
    parser.add_argument('--out', help='Output directory, the model name will be appended as a directory', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="moco_v2")
    parser.add_argument('--transform', help='Type of transform', type=str, default="real")
    parser.add_argument('--scale_a_max', help='Scale maximum intensity', type=float, default=255.0)
    
    # parser.add_argument('--emb_dim', help='Embedding dimension', type=int, default=128)
    # parser.add_argument('--hidden_dim', help='Hidden Embedding dimension', type=int, default=None)
    # parser.add_argument('--compute_feat', help='Remove last MLP Layer', type=bool, default=False)
    # parser.add_argument('--base_encoder', help='What encoder to use', type=str, default='efficientnet_b0')
    # parser.add_argument('--n_clusters', help='Guess number of clusters', type=int, default=64)
    # parser.add_argument('--n_lights', help='Light house', type=int, default=10)


    args = parser.parse_args()

    main(args)
