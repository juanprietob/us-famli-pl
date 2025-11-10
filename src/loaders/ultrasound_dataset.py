import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import ast

from .transforms import ultrasound_transforms as ust

from lightning.pytorch.core import LightningDataModule


from monai.transforms import (    
    LoadImage
)

TAGS_DICT = {
        'AC': -1,            
        'BPD': -1,
        'TCD': -1,
        'FL': -1,
        'HL': -1,
        'CRL': -1,
        'C1': 0,
        'C2': 1,
        'C3': 2,
        'C4': 3,
        'C5': 4,
        'FA1': 5,
        'FA2': 6,
        'L0': 7,
        'L1': 8,
        'L15': 9,
        'L45': 10,
        'M': 11,
        'R0': 12,
        'R1': 13,
        'R15': 14,
        'R45': 15,
        'RTA': 16,
        'NL': 7,
        'NR': 12,
        'IL0': 7,
        'ILO': 7,
        'IRO': 12,
        'IR0': 12,
        'IM': 11,
        'IC1': 0,
        'IC2': 1,
        'ML': 11,
        'MR': 12,
        'L2': 8,
        'NC1': 0,
        'NC2': 1,
        'NC3': 2,        
        'NM': 11, 
        'RTB': 16,
        'RTC': 16,
    }


class USDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=False, class_dict=None, scalar_dict=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel        
        self.class_dict = class_dict
        self.scalar_dict = scalar_dict

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            
            img = sitk.ReadImage(img_path)

            img_np = sitk.GetArrayFromImage(img)
            img_t = torch.tensor(img_np, dtype=torch.float32)

            if img.GetNumberOfComponentsPerPixel() == 1 and self.repeat_channel:
                img_t = img_t.unsqueeze(-1).repeat(1, 1, 3)

            img_t = img_t.permute(2, 0, 1)
                    
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([3, 256, 256]), dtype=torch.float32)

        if(self.transform):
            img_t = self.transform(img_t)            

        ret_dict = {"img": img_t}

        if self.class_column:
            if self.class_dict:
                class_idx = self.class_dict[self.df.iloc[idx][self.class_column]]
                ret_dict["class"] = torch.tensor(class_idx).to(torch.long)
            else:
                ret_dict["class"] = torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            ret_dict["ga"] = torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            if self.scalar_dict:
                scalar = self.scalar_dict[scalar]
            ret_dict["scalar"] = torch.tensor(scalar)

        return ret_dict

class USDatasetV2(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.return_head = return_head
        self.loader = LoadImage(ensure_channel_first=True, reverse_indexing=False)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            # img, head = nrrd.read(img_path, index_order="C")
            # img = img.astype(float)            
            # img = torch.tensor(img, dtype=torch.float32).detach()        
            # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            # img = torch.tensor(img, dtype=torch.float32)
            img = self.loader(img_path)[0]
            img = torch.permute(img, (0, 2, 1))
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([3, 256, 256]), dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        return img

class SimuDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False, target_column=None, target_transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.target_column = target_column
        self.target_transform = target_transform
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                img = img.astype(float)
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)            
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)

        img = img/339
        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        if self.target_column:
            try:
                target_img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.target_column])
                if os.path.splitext(img_path)[1] == ".nrrd":
                    target, head = nrrd.read(target_img_path, index_order="C")
                    # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    target = target.squeeze()
                    target = target[:,:,0]
                    if self.repeat_channel:
                        target = target.unsqueeze(0).repeat(3,1,1)
                else:
                    target = np.array(Image.open(target_img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    if len(img.shape) == 3:                    
                        target = torch.permute(img, [2, 0, 1])[0:3, :, :]
                    else:                    
                        target = target.unsqueeze(0).repeat(3,1,1)            
            except:
                print("Error reading frame: " + target_img_path, file=sys.stderr)
                target = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)
            
            target = target/255
            if(self.transform):
                target = self.transform(target)

            return img, target

        return img

class USDatasetBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column=None, transform=None, id_column=None, max_sweeps=4):
        self.df = df
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_sweeps = max_sweeps

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        if self.id_column:
            df_group = self.df_group.get_group(self.keys[idx])
            ga = float(df_group[self.ga_column].unique()[0])
        
            img = self.create_seq(df_group)
        
            return img, torch.tensor([ga], dtype=torch.float32)
        else:
        
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            try:
                img = sitk.ReadImage(img_path)
                img_np = sitk.GetArrayFromImage(img)

                if(img.GetNumberOfComponentsPerPixel() > 1):                    
                    img_np = img_np[:,:,:,0]

                img_t = torch.from_numpy(img_np).to(torch.float32)
                if self.num_frames > 0:
                    idx = torch.randint(low=0, high=img.shape[0], size=self.num_frames)
                    # idx = torch.randperm(img.shape[0])[:self.num_frames]
                    if self.num_frames == 1:
                        img_t = img_t[idx[0]]
                    else:
                        img_t = img_t[idx]
            except:
                print("Error reading cine: " + img_path)
                if self.num_frames == 1:
                    img_t = torch.zeros(256, 256, dtype=torch.float32)
                else:
                    img_t = torch.zeros(self.num_frames, 256, 256, dtype=torch.float32)
            
            if self.num_frames == 1:
                img_t = img_t.unsqueeze(0).repeat(3,1,1).contiguous()
            else:
                img_t = img_t.unsqueeze(0).repeat(3,1,1,1).contiguous()

            if self.transform:
                img_t = self.transform(img_t)

            if self.ga_column:
                ga = self.df.iloc[idx][self.ga_column]
                return img_t, torch.tensor([ga])

            return img_t

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_sweeps = len(df.index)
        if self.max_sweeps > -1:
            max_sweeps = min(max_sweeps, self.max_sweeps)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_sweeps].sort_index()

        # read all of them
        
        imgs = []

        for idx, row in df.iterrows():
            # try:
            img_path = os.path.join(self.mount_point, row[self.img_column])                

            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)

            if(img.GetNumberOfComponentsPerPixel() > 1):
                img_np = img_np[:,:,:,0]
            
            img_t = img_t.unsqueeze(1).repeat(3,1,1,1).contiguous()
            img_t = torch.from_numpy(img_np).to(torch.float32)

            if self.transform:
                img_t = self.transform(img_t)                
            imgs.append(img_t)
            # except Exception as e:
            #     print(e, file=sys.stderr)
        return torch.cat(imgs)

class USDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column='ga_boe', id_column='study_id', max_seq=-1, transform=None):
        self.df = df
        
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_seq = max_seq

        print(id_column)
        self.df_group = self.df.groupby(id_column)
        print(len(self.df_group.groups.keys()))
        self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        df_group = self.df_group.get_group(self.keys[idx])
        ga = float(df_group[self.ga_column].unique()[0])
        
        img = self.create_seq(df_group)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor([ga], dtype=torch.float32)

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_seq = len(df.index)
        if self.max_seq > -1:
            max_seq = min(max_seq, self.max_seq)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_seq].sort_index()

        # read all of them
        imgs = []
        time_steps = 0 
        for idx, row in df.iterrows():
            try:
                img_path = os.path.join(self.mount_point, row[self.img_column])
                img_np, head = nrrd.read(img_path, index_order="C")

                if self.transform:
                    img_np = self.transform(img_np)

                imgs.append(img_np)
            except Exception as e:
                print(e, file=sys.stderr)

        return np.stack(imgs)

    # def has_all_types(self, df, seqo):
    #     if(seqo[0] == "all"):
    #         return True
    #     seq_found = np.zeros(len(seqo))
    #     for i, t in df["tag"].items():
    #         scan_index = np.where(np.array(seqo) == t)[0]
    #         for s_i in scan_index:
    #             seq_found[s_i] += 1
    #     return np.prod(seq_found) > 0

# class ITKImageDataset(Dataset):
#     def __init__(self, csv_file, transform=None, target_transform=None):
#         self.df = pd.read_csv(csv_file)

#         self.transform = transform
#         self.target_transform = target_transform
#         self.sequence_order = ["all"]

#         self.df = self.df.groupby('study_id').filter(lambda x: has_all_types(x, self.sequence_order))

#         self.df_group = self.df.groupby('study_id')
#         self.keys = list(self.df_group.groups.keys())
#         self.data_frames = []

#     def __len__(self):
#         return len(self.df_group)

#     def __getitem__(self, idx):

#         df_group = self.df_group.get_group(self.keys[idx])

#         seq_np, df = create_seq(df_group, self.sequence_order)
#         ga = df["ga_boe"]
#         # img = self.df.iloc[idx]['file_path']
#         # ga = self.df.iloc[idx]['ga_boe']

#         # reader = ITKReader()
#         # img = reader.read(img)

#         # if self.transform:
#         #     img = self.transform(img)
#         # if self.target_transform:
#         #     ga = self.target_transform(ga)

#         self.data_frames.append(df)

#         return (self.transform(seq_np), np.array([ga]))


class USDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)
        self.mount_point = self.hparams.mount_point
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.img_column = self.hparams.img_column
        self.class_column = self.hparams.class_column
        self.scalar_column = self.hparams.scalar_column
        self.ga_column = self.hparams.ga_column
        self.train_transform = ust.USClassTrainTransforms()
        self.valid_transform = ust.USClassEvalTransforms()
        self.test_transform = ust.USClassEvalTransforms()
        self.drop_last = self.hparams.drop_last
        self.repeat_channel = self.hparams.repeat_channel
        self.target_column = self.hparams.target_column
        if self.hparams.class_dict:
            self.class_dict = json.load(open(self.hparams.class_dict))
        else:
            self.class_dict = None

        if self.hparams.scalar_dict:
            self.scalar_dict = json.load(open(self.hparams.scalar_dict))
        else:
            self.scalar_dict = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads frames")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, required=True, help="Training data csv file path")
        group.add_argument('--csv_valid', type=str, required=True, help="Validation data csv file path")
        group.add_argument('--csv_test', type=str, required=True, help="Test data csv file path")
        group.add_argument('--img_column', type=str, default="file_path")
        group.add_argument('--class_column', type=str, default=None)
        group.add_argument('--ga_column', type=str, default=None)
        group.add_argument('--scalar_column', type=str, default=None)
        group.add_argument('--target_column', type=str, default=None)
        group.add_argument('--repeat_channel', type=int, default=1, help="Repeat single channel to 3 channels")
        group.add_argument('--class_dict', type=str, default=None, help="Dictionary mapping class names to indices")
        group.add_argument('--scalar_dict', type=str, default=None, help="Dictionary mapping class names to scalar values")
        group.add_argument('--batch_size', type=int, default=4, help="Batch size for the train dataloaders")        
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--prefetch_factor', type=int, default=2)
        group.add_argument('--drop_last', type=int, default=False)

        return parent_parser

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel, class_dict=self.class_dict, scalar_dict=self.scalar_dict)
        self.val_ds = USDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel, class_dict=self.class_dict, scalar_dict=self.scalar_dict)
        self.test_ds = USDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel, class_dict=self.class_dict, scalar_dict=self.scalar_dict)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleV2(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last    
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetV2(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform)
        self.val_ds = USDatasetV2(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform)
        self.test_ds = USDatasetV2(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class SimuDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = SimuDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.val_ds = SimuDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.test_ds = SimuDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleBlindSweep(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, num_frames=50, max_sweeps=-1, img_column='uuid_path', ga_column=None, id_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.num_frames = num_frames
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.max_sweeps = max_sweeps

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweep(self.df_train, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column,id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.train_transform)
        self.val_ds = USDatasetBlindSweep(self.df_val, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.valid_transform)
        self.test_ds = USDatasetBlindSweep(self.df_test, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_seq)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        blind_sweeps = [bs for bs, g in batch]
        ga = [g for v, g in batch]    
        
        blind_sweeps = pad_sequence(blind_sweeps, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return blind_sweeps, ga


class USDataModuleVolumes(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, max_seq=5, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column= id_column
        self.max_seq = max_seq
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.train_transform)
        self.val_ds = USDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.valid_transform)
        self.test_ds = USDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_volumes)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def pad_volumes(self, batch):

        volumes = [v for v, g in batch]
        ga = [g for v, g in batch]    
        
        volumes = pad_sequence(volumes, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return volumes, ga

class USZDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USZDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform)
        self.val_ds = USZDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform)
        self.test_ds = USZDataset(self.df_test, self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)
    

class USDatasetBlindSweepWTag(Dataset):
    def __init__(self, df, mount_point = "./", img_column='file_path', ga_column=None, efw_column=None, tag_column=None, class_column=None, frame_column=None, presentation_column=None, transform=None, id_column='study_id', max_sweeps=3, num_frames=96):
        
        self.df = df
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.id_column = id_column
        self.tag_column = tag_column
        self.class_column = class_column
        self.efw_column = efw_column
        self.ga_column = ga_column
        self.frame_column = frame_column
        self.presentation_column = presentation_column
        self.max_sweeps = max_sweeps

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

        self.tags_dict = TAGS_DICT
        self.max_tag = np.max(list(self.tags_dict.values())) + 1
        
        self.presentation_dict = {'cephalic': 0,
            'transverse': 1,
            'breech': 2}
        
        self.class_dict = {
            'AC': 0, 
            'BPD': 1,
            'TCD': 1,
            'FL': 2,
            'HL': 2,
            'CRL': 3
        }

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        if self.id_column:
            df_group = self.df_group.get_group(self.keys[idx])
            
            ret_dict = self.create_seq(df_group)
            
            if self.frame_column:
                row = df_group.iloc[0]
                frame = np.array(ast.literal_eval(row[self.frame_column])).reshape(3, 3)

                ret_dict["frame"] = torch.tensor(frame, dtype=torch.float32)
            
            if self.presentation_column:
                row = df_group.iloc[0]
                presentation = row[self.presentation_column]
                ret_dict["presentation"] = torch.tensor(self.presentation_dict[presentation], dtype=torch.long)

            if self.ga_column:
                row = df_group.iloc[0]
                ga = row[self.ga_column]
                ret_dict["ga"] = torch.tensor(ga, dtype=torch.float32)

            if self.efw_column:
                row = df_group.iloc[0]
                efw = row[self.efw_column]
                ret_dict["efw"] = torch.tensor([efw], dtype=torch.float32)/1000.0

            return ret_dict

        else:
        
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            try:
                img = sitk.ReadImage(img_path)
                img_np = sitk.GetArrayFromImage(img)

                if (img.GetNumberOfComponentsPerPixel() > 1):
                    img_np = img_np[:, :, :, 0]

                img_t = torch.from_numpy(img_np).float()
                if self.num_frames > 0:
                    idx_f = torch.randint(low=0, high=img_t.shape[0], size=(self.num_frames,))
                    idx_f = idx_f.sort().values
                    img_t = img_t[idx_f].contiguous()
                
            except:
                print("Error reading cine: " + img_path)
                if self.num_frames == 1:
                    img_t = torch.zeros(256, 256, dtype=torch.float32)
                else:
                    img_t = torch.zeros(self.num_frames, 256, 256, dtype=torch.float32)
            
            if self.num_frames == 1:
                img_t = img_t.unsqueeze(0).repeat(3,1,1).contiguous()
            else:
                img_t = img_t.unsqueeze(0).repeat(3,1,1,1).contiguous()

            if self.transform:
                img_t = self.transform(img_t)

            ret_dict = {"img": img_t}

            if self.ga_column:
                ga = self.df.iloc[idx][self.ga_column]
                ret_dict["ga"] = torch.tensor([ga])

            if self.efw_column:
                efw = self.df.iloc[idx][self.efw_column]
                ret_dict["efw"] = torch.tensor([efw])/1000.0

            if self.tag_column:
                sweep_tag = self.tags_dict[self.df.iloc[idx][self.tag_column]]
                if sweep_tag == -1:
                    
                    sweep_tag = torch.randint(low=0, high=self.max_tag, size=(1,)).item()
                ret_dict["tag"] = torch.tensor(sweep_tag, dtype=torch.long)

            if self.class_column:
                cl = self.df.iloc[idx][self.class_column]
                ret_dict["class"] = torch.tensor(self.class_dict[cl], dtype=torch.long)

            return ret_dict

    def create_seq(self, df):

        if(self.max_sweeps > -1):
            replace = len(df.index) < self.max_sweeps
            df = df.sample(n=self.max_sweeps, replace=replace)

        ret_dict = {}
        
        imgs = []
        tags = []

        for idx, row in df.iterrows():
            # try:
            img_path = os.path.join(self.mount_point, row[self.img_column])                
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)

            if (img.GetNumberOfComponentsPerPixel() > 1):
                img_np = img_np[:, :, :, 0]

            img_t = torch.from_numpy(img_np).float()
            if self.num_frames > 0:
                idx_f = torch.randint(low=0, high=img_t.shape[0], size=(self.num_frames,))
                idx_f = idx_f.sort().values
                img_t = img_t[idx_f].contiguous()

            img_t = img_t.unsqueeze(0).repeat(3,1,1,1).contiguous()

            if self.transform:
                img_t = self.transform(img_t)
            imgs.append(img_t)

            if self.tag_column:            
                tags.append(self.tags_dict[row[self.tag_column]])
            # except Exception as e:
            #     print(e, file=sys.stderr)

        ret_dict["img"] = torch.stack(imgs)

        if self.tag_column:
            ret_dict["tag"] = torch.tensor(tags, dtype=torch.long)
        
        return ret_dict

class USDataModuleBlindSweepWTag(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)        
        self.train_transform = ust.BlindSweepTrainTransforms()
        self.valid_transform = ust.BlindSweepEvalTransforms()
        self.test_transform = ust.BlindSweepEvalTransforms()        

    staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("USDataModuleBlindSweepWTag")
        
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--batch_size', type=int, default=2)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--ga_column', type=str, default=None)
        group.add_argument('--class_column', type=str, default=None)
        group.add_argument('--frame_column', type=str, default=None)
        group.add_argument('--tag_column', type=str, default="tag")
        group.add_argument('--presentation_column', type=str, default=None)
        group.add_argument('--efw_column', type=str, default=None)
        group.add_argument('--id_column', type=str, default=None)
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--csv_test', type=str, default=None, required=True)
        group.add_argument('--num_frames', type=int, default=128)
        group.add_argument('--num_frames_val', type=int, default=-1)
        group.add_argument('--num_frames_test', type=int, default=-1)
        group.add_argument('--max_sweeps', type=int, default=3)        
        group.add_argument('--drop_last', type=int, default=False)
        group.add_argument('--prefetch_factor', type=int, default=2)

        return parent_parser

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweepWTag(self.df_train, mount_point=self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, ga_column=self.hparams.ga_column, id_column=self.hparams.id_column, frame_column=self.hparams.frame_column, class_column=self.hparams.class_column, presentation_column=self.hparams.presentation_column, efw_column=self.hparams.efw_column, max_sweeps=self.hparams.max_sweeps, transform=self.train_transform, num_frames=self.hparams.num_frames)
        self.val_ds = USDatasetBlindSweepWTag(self.df_val, mount_point=self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, ga_column=self.hparams.ga_column, id_column=self.hparams.id_column, frame_column=self.hparams.frame_column, class_column=self.hparams.class_column, presentation_column=self.hparams.presentation_column, efw_column=self.hparams.efw_column, max_sweeps=-1, transform=self.valid_transform, num_frames=self.hparams.num_frames_val)
        self.test_ds = USDatasetBlindSweepWTag(self.df_test, mount_point=self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, ga_column=self.hparams.ga_column, id_column=self.hparams.id_column, frame_column=self.hparams.frame_column, class_column=self.hparams.class_column, presentation_column=self.hparams.presentation_column, efw_column=self.hparams.efw_column, max_sweeps=-1, transform=self.test_transform, num_frames=self.hparams.num_frames_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last)


class USAnnotatedBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", img_column='file_path', tag_column='tag', frame_column="frame_index", frame_label="annotation_label", id_column='annotation_id', num_frames=64, transform=None):
        
        self.df_frames = df        
        self.mount_point = mount_point
        self.num_frames = num_frames        
        self.img_column = img_column
        self.id_column = id_column
        self.tag_column = tag_column        
        self.frame_column = frame_column
        self.frame_label = frame_label        
        self.transform = transform

        self.df = self.df_frames[[id_column, img_column, tag_column]].drop_duplicates().reset_index(drop=True)

        self.keys = self.df.index

        self.tags_dict = TAGS_DICT

        self.max_tag = np.max(list(self.tags_dict.values())) + 1

        self.frame_labels_dict = {
            'low_visible': 1,
            'high_visible': 2,
            'low_measurable': 3,
            'high_measurable': 4
        }

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        uid = self.df.iloc[idx][self.id_column]
        frames = self.df_frames[self.df_frames[self.id_column] == uid].sort_values(by=self.frame_column).reset_index(drop=True)

        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        sweep_tag = self.tags_dict[self.df.iloc[idx][self.tag_column]]
        if sweep_tag == -1:            
            sweep_tag = torch.randint(low=0, high=self.max_tag, size=(1,)).item()

        sweep_tag_t = torch.tensor(sweep_tag, dtype=torch.int64)

        try:
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)

            if img.GetNumberOfComponentsPerPixel() >  1:
                img_np = img_np[:,:,:,0]

            img_t = torch.tensor(img_np, dtype=torch.float32)

            frame_idx = frames[self.frame_column].values.tolist()
            frame_idx = np.clip(frame_idx, 0, img_t.shape[0]-1)

            frame_labels = frames[self.frame_label].values.tolist()
            frame_labels_idx = [self.frame_labels_dict[lbl] for lbl in frame_labels]

            img_labels_t = torch.zeros(img_t.shape[0], dtype=torch.int64)
            img_labels_t[frame_idx] = torch.tensor(frame_labels_idx, dtype=torch.int64)

            if self.num_frames > 0:
                idx = torch.randint(low=0, high=img_t.shape[0], size=(self.num_frames,))
                idx = idx.sort().values

                img_t = img_t[idx].contiguous()
                img_labels_t = img_labels_t[idx].contiguous()

            img_t = img_t.unsqueeze(0).repeat(3,1,1,1).contiguous()            

        except Exception as e:
            print("Error reading cine: " + img_path)
            print("Error: " + str(e))
            if self.num_frames == 1:
                img_t = torch.zeros(256, 256, dtype=torch.float32)
            elif self.num_frames == -1:
                img_t = torch.zeros(1, 256, 256, dtype=torch.float32)
            else:
                img_t = torch.zeros(self.num_frames, 256, 256, dtype=torch.float32)        

        if self.transform:
            img_t = self.transform(img_t)

        return {"img": img_t, "tag": sweep_tag_t, "class": img_labels_t}
    
class USAnnotatedBlindSweepDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_valid = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)

        self.train_transform = ust.BlindSweepTrainTransforms()
        self.valid_transform = ust.BlindSweepEvalTransforms()
        self.test_transform = ust.BlindSweepEvalTransforms()
        
    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads a series of images to train the blind sweep frame classification Model")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, required=True, help="Training data csv file path")
        group.add_argument('--csv_valid', type=str, required=True, help="Validation data csv file path")
        group.add_argument('--csv_test', type=str, required=True, help="Test data csv file path")
        group.add_argument('--img_column', type=str, default="file_path")
        group.add_argument('--tag_column', type=str, default="tag")
        group.add_argument('--frame_column', type=str, default="frame_index")
        group.add_argument('--frame_label', type=str, default="annotation_label")
        group.add_argument('--id_column', type=str, default="annotation_id")
        group.add_argument('--batch_size', type=int, default=4, help="Batch size for the train dataloaders")
        group.add_argument('--num_frames', type=int, default=64, help="Number of frames to sample from each cine")
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--prefetch_factor', type=int, default=2)
        group.add_argument('--drop_last', type=int, default=False)

        return parent_parser        

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders        
        self.train_ds = USAnnotatedBlindSweep(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=self.hparams.num_frames, transform=self.train_transform)
        self.val_ds = USAnnotatedBlindSweep(self.df_valid, self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=-1, transform=self.valid_transform)
        self.test_ds = USAnnotatedBlindSweep(self.df_test, self.hparams.mount_point, img_column=self.hparams.img_column, tag_column=self.hparams.tag_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=-1, transform=self.test_transform)

    def collate_fn(self, batch):

        if self.hparams.tag_column is not None:
                imgs = torch.cat([item[0] for item in batch], dim=0)
                tags = torch.cat([item[1] for item in batch], dim=0)
                labels = torch.cat([item[2] for item in batch], dim=0)

                return imgs, tags, labels

        else:
            imgs = torch.cat([item[0] for item in batch], dim=0)            
            labels = torch.cat([item[1] for item in batch], dim=0)

            return imgs, labels

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers)


class USAnnotatedBlindSweep2DFrames(Dataset):
    def __init__(self, df, mount_point = "./", img_column='file_path', frame_column="frame_index", frame_label="annotation_label", id_column='annotation_id', num_frames=64, transform=None, df_extra_ac=None):
        
        self.df_frames = df        
        self.mount_point = mount_point
        self.num_frames = num_frames        
        self.img_column = img_column
        self.id_column = id_column         
        self.frame_column = frame_column
        self.frame_label = frame_label        
        self.transform = transform
        self.df_extra_ac = df_extra_ac

        self.df = self.df_frames[[id_column, img_column]].drop_duplicates().reset_index(drop=True)

        self.keys = self.df.index

        self.frame_labels_dict = {
            'low_visible': 1,
            'high_visible': 2,
            'low_measurable': 3,
            'high_measurable': 4
        }

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        uid = self.df.iloc[idx][self.id_column]
        frames = self.df_frames[self.df_frames[self.id_column] == uid].sort_values(by=self.frame_column).reset_index(drop=True)

        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        img = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img)

        if img.GetNumberOfComponentsPerPixel() >  1:
            img_np = img_np[:,:,:,0]

        img_t = torch.tensor(img_np, dtype=torch.float32)

        frame_idx = frames[self.frame_column].to_numpy()
        frame_idx = np.clip(frame_idx, 0, img_t.shape[0]-1)
        frame_idx_t = torch.as_tensor(frame_idx, dtype=torch.long)


        frame_labels = frames[self.frame_label].values.tolist()
        frame_labels_idx = [self.frame_labels_dict[lbl] for lbl in frame_labels]

        img_labels_t = torch.zeros(img_t.shape[0], dtype=torch.long)
        img_labels_t[frame_idx_t] = torch.tensor(frame_labels_idx, dtype=torch.long)

        if self.num_frames > 0:

            idx_pos = torch.randint(low=0, high=frame_idx_t.numel(), size=(self.num_frames//2,))
            pos_sample = frame_idx_t[idx_pos]
            pos_sample = pos_sample.unique()

            frame_idx_neg_t = (img_labels_t == 0).nonzero(as_tuple=True)[0]
            idx_neg = torch.randint(low=0, high=frame_idx_neg_t.numel(), size=(min(self.num_frames//2, pos_sample.numel()),))
            neg_sample = frame_idx_neg_t[idx_neg].unique()

            sample_idx = torch.cat([pos_sample, neg_sample], dim=0)
            sample_idx = sample_idx.sort().values

            img_t = img_t[sample_idx]
            img_labels_t = img_labels_t[sample_idx]            

        img_t = img_t.unsqueeze(1).repeat(1,3,1,1).contiguous()            
        img_labels_t = img_labels_t

        if self.transform:
            img_t = self.transform(img_t)


        if (self.df_extra_ac is not None):
            img_path_ac = os.path.join(self.mount_point, self.df_extra_ac.sample(n=1)[self.img_column].values[0])
            if os.path.exists(img_path_ac):
                img_ac = sitk.ReadImage(img_path_ac)
                img_ac_np = sitk.GetArrayFromImage(img_ac)

                if img_ac.GetNumberOfComponentsPerPixel() >  1:
                    img_ac_np = img_ac_np[:,:,:,0]

                img_ac_t = torch.tensor(img_ac_np, dtype=torch.float32)

                #Sample 8 random frames from the last 10% of the cine
                num_frames_ac = img_ac_t.shape[0]
                start_frame = int(num_frames_ac * 0.9)
                idx_ac = torch.randint(low=start_frame, high=num_frames_ac, size=(5,))
                idx_ac = idx_ac.sort().values

                img_ac_t = img_ac_t[idx_ac]
                img_ac_t = img_ac_t.unsqueeze(1).repeat(1,3,1,1).contiguous()
                img_labels_ac_t = torch.ones(img_ac_t.shape[0])*self.frame_labels_dict['high_measurable']
                if self.transform:
                    img_ac_t = self.transform(img_ac_t)
                img_t = torch.cat([img_t, img_ac_t], dim=0)
                img_labels_t = torch.cat([img_labels_t, img_labels_ac_t], dim=0)            

        return img_t, img_labels_t
    
class USAnnotatedBlindSweep2DFramesDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_valid = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)

        self.df_train_extra = None
        if(self.hparams.use_extra_ac):
            self.df_train_extra = pd.read_csv('/mnt/raid/C1_ML_Analysis/CSV_files/c3_instance_analysis_dataset_ac_train_masked_resampled_256_spc075.csv')

        self.train_transform = ust.BlindSweepTrainTransforms()
        self.valid_transform = ust.BlindSweepEvalTransforms()
        self.test_transform = ust.BlindSweepEvalTransforms()
        
    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads a series of images to train the blind sweep frame classification Model")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, required=True, help="Training data csv file path")
        group.add_argument('--csv_valid', type=str, required=True, help="Validation data csv file path")
        group.add_argument('--csv_test', type=str, required=True, help="Test data csv file path")
        group.add_argument('--use_extra_ac', type=int, default=0, help="Extra sweeps for training e.g.,")        
        group.add_argument('--img_column', type=str, default="file_path")        
        group.add_argument('--frame_column', type=str, default="frame_index")
        group.add_argument('--frame_label', type=str, default="annotation_label")
        group.add_argument('--id_column', type=str, default="annotation_id")
        group.add_argument('--batch_size', type=int, default=4, help="Batch size for the train dataloaders")
        group.add_argument('--num_frames', type=int, default=64, help="Number of frames to sample from each cine")
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--prefetch_factor', type=int, default=2)
        group.add_argument('--drop_last', type=int, default=False)

        return parent_parser        

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders        
        self.train_ds = USAnnotatedBlindSweep2DFrames(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=self.hparams.num_frames, transform=self.train_transform, df_extra_ac=self.df_train_extra)
        self.val_ds = USAnnotatedBlindSweep2DFrames(self.df_valid, self.hparams.mount_point, img_column=self.hparams.img_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=-1, transform=self.valid_transform)
        self.test_ds = USAnnotatedBlindSweep2DFrames(self.df_test, self.hparams.mount_point, img_column=self.hparams.img_column, frame_column=self.hparams.frame_column, frame_label=self.hparams.frame_label, id_column=self.hparams.id_column, num_frames=-1, transform=self.test_transform)

    def collate_fn(self, batch):

        imgs = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.cat([item[1] for item in batch], dim=0)

        return imgs, labels
    
    def train_collate_fn(self, batch):

        imgs, labels = self.collate_fn(batch)

        idx = torch.randperm(imgs.size(0))
        imgs = imgs[idx]
        labels = labels[idx]
        return imgs, labels


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=self.hparams.prefetch_factor, collate_fn=self.train_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate_fn)