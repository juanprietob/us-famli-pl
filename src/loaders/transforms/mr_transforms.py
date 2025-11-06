

import math
import torch
from torch import nn

import monai
from monai import transforms

import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Divide by zero (a_min == a_max)")

class random_slice():
    def __init__(self, size=1):
        self.size = size
    def __call__(self, img, size=1):
        img = img[:,np.random.randint(0, high=img.shape[1], size=(self.size,))]
        return img
    

    
class random_mask():
    def __init__(self, mount_point):


        self.mask_transform = transforms.Compose(
            [                
                transforms.LoadImage(image_only=True, ensure_channel_first=True),
                transforms.RandRotate(range_x=math.pi, prob=1.0, mode="nearest", padding_mode="zeros")
            ]
        )
        self.mount_point = mount_point
        
    def __call__(self, img):                        
        img_mask_np = self.mask_transform(os.path.join(self.mount_point, 'probe_fan_resampled', str(np.random.randint(low=0, high=999)) + ".nrrd"))        
        return img*img_mask_np.squeeze()

class MRDiffusionTrainTransforms:
    def __init__(self, size: int =256, mount_point: str = "", random_slice_size=10, probe_fan_prob=0.5):

        self.train_transform = transforms.Compose(
            [                
                transforms.LoadImage(image_only=True, ensure_channel_first=True),        
                transforms.RandRotate(range_x=math.pi, range_y=math.pi, range_z=math.pi, prob=0.75, mode="nearest"),
                random_slice(size=random_slice_size),
                transforms.ResizeWithPadOrCrop(spatial_size=(-1, 256, 256), mode='constant'),        
                transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.5),        
                transforms.RandLambda(func=random_mask(mount_point), prob=probe_fan_prob),        
                transforms.ScaleIntensityRangePercentiles(lower=0, upper=99.5, b_min=0, b_max=1),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class MRDiffusionEvalTransforms:
    def __init__(self, size: int = 256, mount_point: str = "", random_slice_size=10, probe_fan_prob=0.5):

        self.eval_transform = transforms.Compose(
            [        
                transforms.LoadImage(image_only=True, ensure_channel_first=True),
                random_slice(size=random_slice_size),
                transforms.ResizeWithPadOrCrop(spatial_size=(-1, 256, 256), mode='constant'),                
                transforms.RandLambda(func=random_mask(mount_point), prob=probe_fan_prob),        
                transforms.ScaleIntensityRangePercentiles(lower=0, upper=99.5, b_min=0, b_max=1),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)