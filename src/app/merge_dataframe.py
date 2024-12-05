
import os
import argparse
import glob
import pandas as pd
import numpy as np
import pickle


def concatenate_parquet_files(args):
    # Find all .parquet files in the directory
    parquet_files = glob.glob(os.path.join(args.dir, '*.parquet'))

    # Read each parquet file and store in a list
    dataframes = [pd.read_parquet(file) for file in parquet_files]

    # Concatenate all dataframes into a single dataframe
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Write the output in parquet format    
    concatenated_df.to_parquet(args.out, index=False)    

    if args.feat == 0:
        return
    
    # Read each feature file and store in a list
    features = [pickle.load(open(file.replace('.parquet', '.pickle'), 'rb')) for file in parquet_files]

    # Concatenate all features into a single numpy array
    features = np.concatenate(features).squeeze()

    # Write the output in pickle format    
    pickle.dump(features, open(args.out.replace('.parquet', '.pickle'), 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Merge dataframe')    
    parser.add_argument('--dir', type=str, help='Directory with dataframes', required=True)     
    parser.add_argument('--out', type=str, help='Output directory', required=True)    
    parser.add_argument('--feat', type=int, help='Save features', default=0) 

    args = parser.parse_args()

    concatenate_parquet_files(args)