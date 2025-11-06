import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import USDatasetBlindSweep
from loaders.transforms.ultrasound_transforms import USClassEvalTransforms

from nets import classification
from nets.classification import TimeDistributed

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    NN = getattr(classification, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    model = model.eval().cuda()

    if args.extract_features:
        model.extract_features = True

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(args.csv)
    else:        
        df_test = pd.read_parquet(args.csv)

    test_ds = USDatasetBlindSweep(df_test, img_column=args.img_column, mount_point=args.mount_point, transform=USClassEvalTransforms())

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

        predictions = []
        probs = []
        features = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, X in pbar:            
            X = X.cuda().contiguous()
            X = X.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            X = X.view(-1, X.shape[2], X.shape[3], X.shape[4])  # (B*T, C, H, W)            
            if args.extract_features:        
                pred, x_f = model(X)    
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
            
            probs.append(pred.cpu().numpy())            
            pred = torch.argmax(pred, dim=1).cpu().numpy()            
            # pbar.set_description("prediction: {pred}".format(pred=pred))
            predictions.append(pred)

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(out_dir, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:        
        df_test.to_parquet(os.path.join(out_dir, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)

    
    pickle.dump(predictions, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_predictions.pickle")), 'wb'))
    pickle.dump(probs, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_features.pickle")), 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--extract_features', type=int, help='Extract the features', default=0)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="file_path")
    parser.add_argument('--nn', type=str, help='Neural network architecture', default="EfficientNet")
    parser.add_argument('--model', help='Model path for prediction', type=str, default='/mnt/raid/C1_ML_Analysis/train_output/classification/extract_frames_blind_sweeps_c1_30082022_wscores_train_train_sample_clean_feat/epoch=9-val_loss=0.27.ckpt')    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred_class")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)

    args = parser.parse_args()

    main(args)
