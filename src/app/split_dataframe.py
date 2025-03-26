import dask.dataframe as dd
import os
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Split dataframe into smaller chunks')    
    parser.add_argument('--csv', type=str, help='CSV file to split', required=True)    
    parser.add_argument('--n', type=int, help='number of splits', required=True)    
    parser.add_argument('--out', type=str, help='Output directory', required=True)    

    args = parser.parse_args()
    
    ddf = dd.read_parquet(args.csv)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    ddf.repartition(npartitions=args.n).to_parquet(args.out)