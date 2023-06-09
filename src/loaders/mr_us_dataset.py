
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


import pytorch_lightning as pl


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class StackDataset(Dataset):
    def __init__(self, dataset, stack_slices=10, shuffle_df=False):        
        self.dataset = dataset
        if shuffle_df:
            self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)
        self.stack_slices = stack_slices       

    def __len__(self):
        return len(self.dataset)//self.stack_slices

    def __getitem__(self, idx):
        
        start_idx = idx*self.stack_slices

        return torch.stack([self.dataset[idx] for idx in range(start_idx, start_idx + self.stack_slices)], dim=1)

class MRDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', id_column='study_id', transform=None):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        return self.transform(img_path)

class MRDataModuleVolumes(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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
        self.train_ds = MRDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.train_transform)
        self.val_ds = MRDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.valid_transform)
        self.test_ds = MRDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def arrange_slices(self, batch):
        batch = torch.cat(batch, axis=1).permute(dims=(1,0,2,3))
        idx = torch.randperm(batch.shape[0])
        return batch[idx]

class MRUSDataModule(pl.LightningDataModule):
    def __init__(self, mr_dataset_train, mr_dataset_val, us_dataset_train, us_dataset_val, batch_size=4, num_workers=4):
        super().__init__()

        self.mr_dataset_train = mr_dataset_train
        self.us_dataset_train = us_dataset_train

        self.mr_dataset_val = mr_dataset_val
        self.us_dataset_val = us_dataset_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = ConcatDataset(self.mr_dataset_train, self.us_dataset_train)
        self.val_ds = ConcatDataset(self.mr_dataset_val, self.us_dataset_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.arrange_slices)    

    def arrange_slices(self, batch):
        mr_batch = [mr for mr, us in batch]
        us_batch = [us for mr, us in batch]        
        mr_batch = torch.cat(mr_batch, axis=1).permute(dims=(1,0,2,3))
        us_batch = torch.cat(us_batch, axis=1).permute(dims=(1,0,2,3))        
        return mr_batch[torch.randperm(mr_batch.shape[0])], us_batch