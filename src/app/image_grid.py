
import os
import sys

# module_path = os.path.abspath(os.path.join(__file__, '..', '..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from loaders.ultrasound_dataset import USDataset
# from transforms.ultrasound_transforms import Moco2TrainTransforms, Moco2EvalTransforms, AutoEncoderTrainTransforms

# from torchvision import transforms

# import plotly.express as px
# import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import SimpleITK as sitk
import numpy as np

import pickle

import argparse
from mpl_toolkits.axes_grid1 import ImageGrid
import math

def sample_balanced_data(df: pd.DataFrame, class_column='pred_class', min_samples=10, random_state=64):
    # Determine the smallest class size in the 'pred_class' column
    min_count = min(min_samples, df[class_column].value_counts().min())
    
    # Initialize a list to hold indices of the samples from each class
    sampled_indices = []
    
    # Group the DataFrame by 'pred_class' and sample min_count rows from each group
    for label, group in df.groupby(class_column):
        sampled_group = group.sample(n=min_count, random_state=random_state)
        sampled_indices.extend(sampled_group.index.tolist())
    
    # Subset the DataFrame and features array using the sampled indices
    balanced_df = df.loc[sampled_indices].reset_index(drop=True)
    
    return balanced_df


def main(args):

    test_df = pd.read_parquet(args.csv)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    for cl in range(test_df[args.pred_column].min(), test_df[args.pred_column].max() + 1):

        filtered_df = test_df.query('{pred_column} == {cl}'.format(pred_column=args.pred_column, cl=cl)).reset_index(drop=True)

        if len(filtered_df) > 0:
        
            filtered_df = filtered_df.sample(n=args.row*args.col)

            imgs = []    

            for idx, row in filtered_df.iterrows():
                img_path = row[args.img_column]
                img_path = os.path.join(args.mount_point, img_path)

                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

                imgs.append(img)

            if len(imgs) > 0:

                fig = plt.figure(figsize=args.fig_size)
                
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(args.row, args.col)
                    )

                for ax, im in zip(grid, imgs):
                    ax.imshow(im, cmap='gray')

                out_name = os.path.join(args.out, f"{cl}.png")
                print("Writing:", out_name)
                fig.savefig(out_name) 
                plt.close(fig)
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create image grids')    
    parser.add_argument('--csv', type=str, help='CSV file with cluster prediction', required=True)
    parser.add_argument('--mount_point', type=str, help='Dataset mount point', default="./")
    parser.add_argument('--pred_column', type=str, help='Prediction/class column', default="pred_cluster")
    parser.add_argument('--img_column', type=str, help='Image column', default="img_path")
    parser.add_argument('--row', type=int, help='number of rows', default=10)
    parser.add_argument('--col', type=int, help='number of columns', default=10)
    parser.add_argument('--fig_size', nargs="+", type=int, help='Figure size', default=[48, 48])   
    parser.add_argument('--out', type=str, help='Output folder', default="./")
         

    args = parser.parse_args()

    main(args)