import argparse
import os
import sys
import pandas as pd
import numpy as np 

import torch
from loaders.ultrasound_dataset import USDataset
from transforms.ultrasound_transforms import DiffusionEvalTransforms, DiffusionV2EvalTransforms
from torch.utils.data import DataLoader
from nets import diffusion

import nrrd
from tqdm import tqdm
from pathlib import Path

def main(args):

    if args.dir:
        df_test = []
        for fn in Path(args.dir).rglob('*.nrrd'):
            df_test.append({args.img_column: fn.as_posix()})
        df_test = pd.DataFrame(df_test)
    else:
        if(os.path.splitext(args.csv)[1] == ".csv"):        
            df_test = pd.read_csv(args.csv)
        else:        
            df_test = pd.read_parquet(args.csv)


    NN = getattr(diffusion, args.nn)
    model = NN.load_from_checkpoint(args.model)
    model.eval()
    model.cuda()
    
    valid_transform = DiffusionEvalTransforms(args.height)

    test_ds = USDataset(df_test, args.mount_point, img_column=args.img_column, transform=valid_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, X in pbar:
            X = X.cuda(non_blocking=True)
            fname = df_test.loc[idx][args.img_column]
            if args.csv_root_path is not None:
                out_fname = fname.replace(args.csv_root_path, args.out)
            else:
                out_fname = os.path.join(args.out, fname)

            out_dir = os.path.dirname(out_fname)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if not os.path.exists(out_fname) or args.ow:
                try:
                    X_hat = model(X)
                    X_hat = X_hat[0].cpu().numpy()
                    pbar.set_description(out_fname)
                    nrrd.write(out_fname, X_hat, index_order='C')
                except:
                    print("ERROR:", fname, file=sys.stderr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DiffusionAE predict')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="Diffusion_AE")
    input_group.add_argument('--model', help='Model to predict', type=str, required=True)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)    
    input_group.add_argument('--prefetch_factor', help='Number of prefectch for loading', type=int, default=2)    
    
    input_data = input_group.add_mutually_exclusive_group(required=True)
    input_data.add_argument('--dir', type=str, help='Directory with NRRD files')
    input_data.add_argument('--csv', type=str, help='Test CSV')
    input_group.add_argument('--csv_root_path', type=str, default=None, help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
    
    input_group.add_argument('--height', type=int, default=64, help='Resize image transform')
    input_group.add_argument('--img_column', type=str, default='img_path', help='Image column name in the csv')
    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--ow', help='Overwrite', type=int, default=0)

    args = parser.parse_args()

    main(args)
