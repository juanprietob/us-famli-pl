import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
    img = sitk.ReadImage(args.img)
    img_np = sitk.GetArrayFromImage(img).astype(args.type)

    if (args.squeeze):
        img_np = img_np.squeeze()
    
    print(img_np.shape)
    out_img = sitk.GetImageFromArray(img_np)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetOrigin(img.GetOrigin())
    # out_img.SetDirection(img.GetDirection())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.out)
    writer.UseCompressionOn()
    writer.Execute(out_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Input image', required=True)
    parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
    parser.add_argument('--type', type=str, help='Output type', default="ubyte")
    parser.add_argument('--squeeze', type=int, help='Squeeze image dimensions if they are 1', default=0)

    args = parser.parse_args()

    main(args)