import SimpleITK as sitk
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from multiprocessing import Pool, cpu_count
import pandas as pd

class Split(object):
    def __init__(self, args):
        self.args = args
        
    def split_fn(self, img_fn):
        
        img = sitk.ReadImage(img_fn)
        img_np = sitk.GetArrayFromImage(img).astype(self.args.type)

        
        out_dir_fn = os.path.splitext(img_fn)[0]
        if self.args.csv_root:
            out_dir_fn = out_dir_fn.replace(self.args.csv_root, "")

        out_dir = os.path.join(self.args.out, out_dir_fn)
            
        if os.path.exists(out_dir) and not self.args.ow:
            # skip if output exists
            return

        if img.GetDimension() == 3: 

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        
            for idx, frame_np in enumerate(img_np):

                if len(frame_np.shape) == 2:
                    out_img = sitk.GetImageFromArray(frame_np)
                elif len(frame_np.shape) == 3:
                    out_img = sitk.GetImageFromArray(frame_np, isVector=True)
                else:
                    print("4D image?", file=sys.stderr)
                    raise 

                out_img.SetSpacing(img.GetSpacing()[0:2])
                out_img.SetOrigin(img.GetOrigin()[0:2])        

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(out_dir, str(idx) + ".nrrd"))
                writer.UseCompressionOn()
                writer.Execute(out_img)


def main(args):

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.img:
        Split(args).split_fn(args.img)
    else:
        
        split = Split(args)

        df = pd.read_csv(args.csv)
        filenames = df[args.img_column].tolist()

        if args.use_multi:
            process_map(split.split_fn, filenames, max_workers=cpu_count(), chunksize=1)
        else:
            split = Split(args)
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                img_fn = row[args.img_column]
                split.split_fn(img_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--img', type=str, help='Input image')
    input_group.add_argument('--csv', type=str, help='Input csv file')
    parser.add_argument('--csv_root', type=str, help='Remove this root from filename. The rest will be used to create the output directory', default=None)
    parser.add_argument('--img_column', type=str, help='Input column name if using csv flag', default="file_path")
    parser.add_argument('--out', type=str, help='Output directory', required=True)
    parser.add_argument('--type', type=str, help='Output type', default="ubyte")
    parser.add_argument('--use_multi', type=int, help='Use multi processing', default=0)
    parser.add_argument('--ow', type=int, help='Overwrite output', default=0)

    args = parser.parse_args()

    main(args)