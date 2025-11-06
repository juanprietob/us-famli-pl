
import math
import torch
from torch import nn
from torchvision.transforms import v2

# from pl_bolts.v2.dataset_normalizations import (
#     imagenet_normalization
# )

import monai
from monai.transforms import (        
    EnsureChannelFirst,
    RepeatChannel,
    Compose,    
    RandFlip,
    RandRotate,
    CenterSpatialCrop,
    ScaleIntensityRange,
    ScaleIntensity,
    ToTensor, 
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    BorderPad,
    RandSpatialCrop,
    NormalizeIntensity,
    RandAffined,
    EnsureChannelFirstd,
    LoadImaged
)

from monai import transforms as monai_transforms




### TRANSFORMS
class SaltAndPepper:    
    def __init__(self, prob=0.2):
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x
    
class NoiseLevelTransform:
    """
    Adds random Gaussian noise to `x`, scaled per element along a chosen dimension.
    """
    def __init__(self, dim=-1, min_scale=0.0, max_scale=0.5, noise_std=1.0):
        """
        Args:
            dim (int): Dimension along which to vary noise level (e.g. -1=None, 0=batch, 1=frames, 2=channels).
            min_scale (float): Minimum blending factor for noise (0=no noise).
            max_scale (float): Maximum blending factor for noise (1=full noise).
            noise_std (float): Standard deviation of Gaussian noise.
        """
        self.dim = dim
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_std = noise_std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_std

        # Create broadcastable noise scaling
        scale_shape = 1
        if self.dim != -1:
            scale_shape = [1] * x.ndim
            scale_shape[self.dim] = x.shape[self.dim]
        scale = torch.empty(scale_shape, device=x.device).uniform_(self.min_scale, self.max_scale)

        return (1.0 - scale) * x + scale * noise

# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = v2.Compose(
#             [
#                 # v2.RandomResizedCrop(height, scale=(0.2, 1.0)),
#                 # v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
#                 # v2.RandomGrayscale(p=0.2),
#                 v2.Pad(64),
#                 v2.RandomCrop(height),
#                 v2.RandomHorizontalFlip(),
#                 v2.RandomApply([SaltAndPepper(0.05)]),
#                 v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
#                 # v2.RandomRotation(180),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k
# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = v2.Compose(
#             [
#                 v2.RandomHorizontalFlip(),
#                 # v2.RandomRotation(180),
#                 v2.Pad(32),
#                 v2.RandomCrop(height),
#                 v2.RandomApply([SaltAndPepper(0.05)]),
#                 v2.RandomApply([v2.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])], p=0.8),
#                 v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k

class Moco2TrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
                v2.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # v2.RandomApply([SaltAndPepper(0.05)]),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class Moco2EvalTransforms:
    """Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):

        self.eval_transform = v2.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.RandomCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class Moco2TestTransforms:
    """Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):

        self.test_transform = Moco2EvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

class SimCLRTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
                # v2.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                # v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # v2.RandomApply([SaltAndPepper(0.05)]),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.Pad(64),
                v2.RandomCrop(height),
                GaussianNoise()
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimCLREvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = v2.Compose(
            [
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k


class SimCLRTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = SimCLREvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

class SimTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
                    v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    ])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimTrainTransformsV2:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.RandomRotation(180), v2.Pad(32), v2.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

#additional transforms
class SimTrainTransformsV3:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.RandomRotation(180), v2.Pad(32), v2.RandomCrop(height)]),
                v2.Sobel(3)
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k


class SimTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = SimEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

# class USClassTrainTransforms:
#     def __init__(self, height: int = 128):

#         self.train_transform = v2.Compose(
#             [
#                 ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
#                 v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
#                 v2.RandomHorizontalFlip(),
#                 v2.RandomChoice([
#                     v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
#                     v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
#                     ])
#             ]
#         )

#     def __call__(self, inp):
#         return self.train_transform(inp)
class USClassTrainTransforms:
    def __init__(self, size: int = 256):
        
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(size)]),
                    v2.RandomResizedCrop(size=size, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                ]),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
                # NoiseLevelTransform(),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)



class USClassEvalTransforms:

    def __init__(self, size=256):

        self.test_transform = v2.Compose(
            [   
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(size),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __call__(self, inp):
        inp = self.test_transform(inp)
        return inp

class USTrainTransforms:
    def __init__(self, height: int = 128):

        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
                    v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class BlindSweepTrainTransforms:
    def __init__(self):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class BlindSweepEvalTransforms:
    def __init__(self):

        self.eval_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class RandomFrames:
    def __init__(self, num_frames=50):
        self.num_frames=num_frames
    def __call__(self, x):
        if self.num_frames > 0:
            idx = torch.randint(x.size(0), (self.num_frames,))
            x = x[idx]        
        return x

class NormI:
    def __init__(self):
        self.subtrahend=torch.tensor((0.485, 0.456, 0.406))
        self.divisor=torch.tensor((0.229, 0.224, 0.225))
    def __call__(self, x):        
        sub = self.subtrahend.to(x.device, dtype=x.dtype)[..., None, None, None]
        div = self.divisor.to(x.device, dtype=x.dtype)[..., None, None, None]
        return (x - sub) / div
    


class USTrainGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.train_transform = v2.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                BorderPad(spatial_border=[-1, 32, 32]),
                RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
                v2.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class USEvalGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.eval_transform = v2.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                v2.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class USEvalSimTransforms:
    def __init__(self, height: int = 256, repeat_channel=3, scale_a_max=255.0):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                v2.CenterCrop(height),                                
                ScaleIntensityRange(a_min=0.0, a_max=scale_a_max, b_min=0.0, b_max=1.0),
                RepeatChannel(repeat_channel)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class USEvalRealTransforms:
    def __init__(self, height: int = 256, repeat_channel=3):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=2),                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class USEvalTransforms:

    def __init__(self, size=256, unsqueeze=False):

        self.test_transform = v2.Compose(
            [                
                v2.CenterCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                # imagenet_normalization(),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        self.unsqueeze = unsqueeze

    def __call__(self, inp):
        inp = self.test_transform(inp)
        if self.unsqueeze:
            return inp.unsqueeze(dim=0)
        return inp

class US3DTrainTransforms:

    def __init__(self, size=128):
        # image augmentation functions        
        self.train_transform = Compose(
            [
                AddChannel(),                
                RandFlip(prob=0.5),
                RandRotate(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros'),
                CenterSpatialCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RandAdjustContrast(prob=0.5),
                RandGaussianNoise(prob=0.5),
                RandGaussianSmooth(prob=0.5)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)


class US3DEvalTransforms:

    def __init__(self, size=128):

        self.test_transform = v2.Compose(
            [
                AddChannel(),
                CenterSpatialCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)
            ]
        )

    def __call__(self, inp):        
        return self.test_transform(inp)


class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(0.1)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EffnetDecodeTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),                
                v2.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                GaussianNoise(),
                v2.GaussianBlur(5, sigma=(0.1, 2.0)),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class EffnetDecodeEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class EffnetDecodeTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = EffnetDecodeEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)


class AutoEncoderTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                # v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
                    v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    # v2.RandomResizedCrop(size=height, scale=(0.9, 1.0), ratio=(0.9, 1.1))
                    ])
                # v2.RandomRotation(30),
                # v2.RandomResizedCrop(size=height, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            ]
        )
        # self.train_transform = v2.Compose(
        #     [
        #         ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
        #         v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
        #         v2.RandomHorizontalFlip(),
        #         v2.RandomApply([v2.RandomRotation(180)]),
        #         v2.RandomApply([v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))]),
        #     ]
        # )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class AutoEncoderEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class AutoEncoderTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = AutoEncoderEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)



class DiffusionTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
                    v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.Resize(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class DiffusionV2TrainTransforms:
    def __init__(self, height: int = 64):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                FirstChannelOnly(),
                v2.Resize(height, antialias=True),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.RandomHorizontalFlip(),
                # v2.RandomRotation(180)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class DiffusionV2EvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = v2.Compose(
            [                
                FirstChannelOnly(),
                v2.Resize(height, antialias=True),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class LabelTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [   
                ToTensor(),             
                v2.RandomHorizontalFlip(),
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(30), v2.Pad(32), v2.RandomCrop(height)]),
                    v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class LabelEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = v2.Compose(
            [
                ToTensor(),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class MustTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                v2.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class MustEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class FirstChannelOnly:
    def __call__(self, inp):
        return inp[0:1]

class RealUSTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=2),
                FirstChannelOnly(),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class RealEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=2),
                FirstChannelOnly(),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                v2.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class DiffusionTrainTransformsPaired:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel')
                # ScaleIntensity()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransformsPaired:
    def __init__(self, height: int = 256):

        self.eval_transform = v2.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel')
                # ScaleIntensity()
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)





class ZSample:
    def __call__(self, x):   
        z_mu = x['z_mu']
        z_sigma = x['z_sigma']
        return self.sampling(z_mu, z_sigma)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
            """
            From the mean and sigma representations resulting of encoding an image through the latent space,
            obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
            adding the mean.

            Args:
                z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
                z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

            Returns:
                sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
            """
            eps = torch.randn_like(z_sigma)
            z_vae = z_mu + eps * z_sigma
            return torch.tanh(z_vae)

class ZGanTrainTransforms:
    def __init__(self, height: int = 64):

        # image augmentation functions
        self.train_transform = v2.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),  
                ZSample(),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.RandomRotation(180), v2.Pad(8), v2.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class ZGanEvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = v2.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),
                ZSample()
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



    
