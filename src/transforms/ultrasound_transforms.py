
import math
import torch
from torch import nn
from torchvision import transforms

# from pl_bolts.transforms.dataset_normalizations import (
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

# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = transforms.Compose(
#             [
#                 # transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
#                 # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
#                 # transforms.RandomGrayscale(p=0.2),
#                 transforms.Pad(64),
#                 transforms.RandomCrop(height),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomApply([SaltAndPepper(0.05)]),
#                 transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
#                 # transforms.RandomRotation(180),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k
# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(),
#                 # transforms.RandomRotation(180),
#                 transforms.Pad(32),
#                 transforms.RandomCrop(height),
#                 transforms.RandomApply([SaltAndPepper(0.05)]),
#                 transforms.RandomApply([transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])], p=0.8),
#                 transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k

class Moco2TrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomApply([SaltAndPepper(0.05)]),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
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

        self.eval_transform = transforms.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomCrop(height)
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
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                # transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                # transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomApply([SaltAndPepper(0.05)]),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.Pad(64),
                transforms.RandomCrop(height),
                GaussianNoise()
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimCLREvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                transforms.CenterCrop(height)
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
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
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
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(32), transforms.RandomCrop(height)])
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
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(32), transforms.RandomCrop(height)]),
                transforms.Sobel(3)
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
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

#         self.train_transform = transforms.Compose(
#             [
#                 ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
#                 transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomChoice([
#                     transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
#                     transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
#                     ])
#             ]
#         )

#     def __call__(self, inp):
#         return self.train_transform(inp)
class USClassTrainTransforms:
    def __init__(self, height: int = 256):

        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class USClassEvalTransforms:

    def __init__(self, size=256, unsqueeze=False):

        self.test_transform = transforms.Compose(
            [   
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(size),
            ]
        )
        self.unsqueeze = unsqueeze

    def __call__(self, inp):
        inp = self.test_transform(inp)
        if self.unsqueeze:
            return inp.unsqueeze(dim=0)
        return inp

class USTrainTransforms:
    def __init__(self, height: int = 128):

        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)



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
        return (x - self.subtrahend)/self.divisor
    


class USTrainGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.train_transform = transforms.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                BorderPad(spatial_border=[-1, 32, 32]),
                RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
                transforms.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class USEvalGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.eval_transform = transforms.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                transforms.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class USEvalTransforms:

    def __init__(self, size=256, unsqueeze=False):

        self.test_transform = transforms.Compose(
            [                
                transforms.CenterCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                # imagenet_normalization(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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

        self.test_transform = transforms.Compose(
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
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),                
                transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                GaussianNoise(),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class EffnetDecodeEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
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
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    # transforms.RandomResizedCrop(size=height, scale=(0.9, 1.0), ratio=(0.9, 1.1))
                    ])
                # transforms.RandomRotation(30),
                # transforms.RandomResizedCrop(size=height, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            ]
        )
        # self.train_transform = transforms.Compose(
        #     [
        #         ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
        #         transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomApply([transforms.RandomRotation(180)]),
        #         transforms.RandomApply([transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))]),
        #     ]
        # )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class AutoEncoderEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
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
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class DiffusionV2TrainTransforms:
    def __init__(self, height: int = 64):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                FirstChannelOnly(),
                transforms.Resize(height),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionV2EvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = transforms.Compose(
            [                
                FirstChannelOnly(),
                transforms.Resize(height),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class LabelTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [   
                ToTensor(),             
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(30), transforms.Pad(32), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class LabelEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class MustTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class MustEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
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
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=2),
                FirstChannelOnly(),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class RealEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=2),
                FirstChannelOnly(),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class DiffusionTrainTransformsPaired:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel')
                # ScaleIntensity()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransformsPaired:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
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
        self.train_transform = transforms.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),  
                ZSample(),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(8), transforms.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class ZGanEvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = transforms.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),
                ZSample()
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



    
