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

            if self.args.random > 0:
                frame_indices = np.random.choice(img_np.shape[0], min(self.args.random, img_np.shape[0]), replace=False)
            else:
                frame_indices = range(img_np.shape[0])

            for idx, frame_np in enumerate(img_np):

                if (self.args.last_frame == 1 and idx == img_np.shape[0] - 1) or (idx in frame_indices and self.args.last_frame == 0):

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
            cpu_c = args.cpu_count if args.cpu_count else cpu_count()
            process_map(split.split_fn, filenames, max_workers=cpu_c, chunksize=1)
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
    parser.add_argument('--last_frame', type=int, help='Last frame onely to split', default=0)
    parser.add_argument('--random', type=int, help='Random frames to split', default=0)
    parser.add_argument('--out', type=str, help='Output directory', required=True)
    parser.add_argument('--type', type=str, help='Output type', default="ubyte")
    parser.add_argument('--use_multi', type=int, help='Use multi processing', default=0)
    parser.add_argument('--cpu_count', type=int, help='Number of CPUs to use for multiprocessing', default=None)
    parser.add_argument('--ow', type=int, help='Overwrite output', default=0)

    args = parser.parse_args()

    main(args)