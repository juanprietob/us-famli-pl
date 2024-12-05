import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import USDataset
from transforms.ultrasound_transforms import USClassEvalTransforms
from nets.classification import EfficientNet

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    
    model = EfficientNet.load_from_checkpoint(args.model)
    model.eval()
    model.cuda()

    if args.extract_features:
        model.extract_features = True

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(args.csv)
    else:        
        df_test = pd.read_parquet(args.csv)

    use_class_column = False
    if args.class_column is not None and args.class_column in df_test.columns:
        use_class_column = True

    if use_class_column:

        #unique_classes = np.sort(np.unique(df_test[args.class_column]))    

        #class_replace = {}
        #for cn, cl in enumerate(unique_classes):
        #    class_replace[cl] = cn
        #print(unique_classes, class_replace)

        #df_test[args.class_column] = df_test[args.class_column].replace(class_replace)

        test_ds = USDataset(df_test, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, transform=USClassEvalTransforms(channel_dim=args.channel_dim, repeat_channel=args.repeat_channel))    
    else:

        test_ds = USDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, transform=USClassEvalTransforms(channel_dim=args.channel_dim, repeat_channel=args.repeat_channel))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

        predictions = []
        probs = []
        features = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, X in pbar: 
            if use_class_column:
                X, Y = X
            X = X.cuda().contiguous()   
            if args.extract_features:        
                pred, x_f = model(X)    
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
            probs.append(pred.cpu().numpy())        
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            # pbar.set_description("prediction: {pred}".format(pred=pred))
            predictions.append(pred)
            

    df_test[args.pred_column] = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probs, axis=0)    


    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if use_class_column:
        print(classification_report(df_test[args.class_column], df_test[args.pred_column]))

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:        
        df_test.to_parquet(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)

    

    pickle.dump(probs, open(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--extract_features', type=int, help='Extract the features', default=0)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="uuid_path")
    parser.add_argument('--class_column', type=str, help='Column name in the csv file with classes', default=None)
    parser.add_argument('--channel_dim', help='Where is the channel in the images', default=-1)
    parser.add_argument('--repeat_channel', type=int, help='Repeat the channel', default=-1)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred_class")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_b0")

    args = parser.parse_args()

    main(args)
