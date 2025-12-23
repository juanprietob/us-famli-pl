import os
import time
import pickle
import os
import time
from collections import Counter
import re
import math

import pandas as pd
import numpy as np
#import itk
import nrrd

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
from torchvision import transforms as T
import torchvision.models as models
#from torchvision.transforms import ToTensor
from torchvision.transforms import Pad
from torchvision.transforms import Resize, ToPILImage

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data.distributed


dataset_dir = "/mnt/raid/C1_ML_Analysis/"
output_mount = "/mnt/famli_netapp_shared/C1_ML_Analysis/"


batch_size = 1

model_date_version = '251206'
model_path = os.path.join(output_mount, 'train_out/model/model_efw_' + model_date_version + '.pt')

output_dir = f"/mnt/raid/C1_ML_Analysis/test_output/fetal_biometry/efw/{model_date_version}/"

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
error_cines_list = []

class ITKImageDatasetByID(Dataset):
    def __init__(self, df, ground_truth, mount_point, cache=False, 
                num_sample_sweeps=None, num_random_sample_frames=None, num_grid_sample_frames=None, device='cpu'):
        self.df = df
        self.mount = mount_point
        self.ground_truth = ground_truth
        ground_truth_map = self.df[["study_id", 'dataset', ground_truth]].drop_duplicates().reset_index(drop=True)
        self.study_ids = ground_truth_map.study_id.values
        self.targets = ground_truth_map.efw_gt.to_numpy()
        self.ground_truth_map = ground_truth_map
        self.cache = cache
        self.data_map = dict()
        self.device = device
        self.num_sample_sweeps = num_sample_sweeps
        self.num_random_sample_frames = num_random_sample_frames
        self.num_grid_sample_frames = num_grid_sample_frames

    def __len__(self):
        return len(self.study_ids)

    def resample_time_grid_sample(self, seq: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        seq: (T, H, W) grayscale frames (float tensor recommended)
        returns: (target_len, H, W)
        """
        assert seq.dim() == 3, "Expected (T, H, W)"
        T, H, W = seq.shape
        device, dtype = seq.device, seq.dtype

        # Build normalized coordinate grids in [-1, 1]
        gx = torch.linspace(-1, 1, W, device=device, dtype=dtype)   # X (width, W)
        gy = torch.linspace(-1, 1, H, device=device, dtype=dtype)   # Y (height, H)
        gz = torch.linspace(-1, 1, target_len, device=device, dtype=dtype)  # Z (time, T')

        # Meshgrid over (Z, Y, X)
        Z, Y, X = torch.meshgrid(gz, gy, gx, indexing='ij')  # shapes (T', H, W)

        # grid_sample expects the last dim as (x, y, z) for 5D input
        grid = torch.stack([X, Y, Z], dim=-1)  # (T', H, W, 3)

        # Prepare input as 5D volume: (N=1, C=1, D=T, H, W)
        inp = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)

        # Trilinear sampling over the (D,H,W) volume; align_corners=True preserves endpoints mapping
        out = nn.functional.grid_sample(
            inp, grid.unsqueeze(0),  # grid: (1, T', H, W, 3)
            mode='bilinear',         # trilinear in 5D
            padding_mode='border',
            align_corners=True
        )
        return out.squeeze(0).squeeze(0)  # (T', H, W)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id == study_id,:].reset_index(drop=True)
        #shuffle rows
        if self.num_sample_sweeps:
            relevant_rows = relevant_rows.sample(n=self.num_sample_sweeps, replace=True).reset_index(drop=True)
        ga = relevant_rows.loc[0, self.ground_truth]
        if self.num_random_sample_frames:
            im_array = torch.zeros((relevant_rows.shape[0], self.num_random_sample_frames, 1, 256, 256))
        elif self.num_grid_sample_frames:
            im_array = torch.zeros((relevant_rows.shape[0], self.num_grid_sample_frames, 1, 256, 256))
        else:
            raise NotImplementedError("Both num_random_sample_frames and num_grid_sample_frames cannot be None")
        for i, row in relevant_rows.iterrows():
            img_path = row['file_path']
            try:
                if self.cache and (img_path in self.data_map):
                    img = self.data_map[img_path]
                else:
                    img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                    if self.cache:
                        self.data_map[img_path] = img
                img = torch.tensor(img, dtype=torch.float, device=self.device)
                if (len(img.shape) == 4):
                    img = img[:,:,:,0]
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(1, 256, 256)
            if self.num_random_sample_frames:
                #idx = torch.randint(img.size(0), (self.num_random_sample_frames,))
                if self.num_random_sample_frames < img.size(0):
                    idx = np.sort(np.random.choice(img.size(0), self.num_random_sample_frames, replace=False))
                else:
                    idx = np.sort(np.tile(np.arange(img.size(0)), int(math.ceil(self.num_random_sample_frames/img.size(0))))[:self.num_random_sample_frames])
                img = img[idx]
            if self.num_grid_sample_frames:
                img = self.resample_time_grid_sample(img, self.num_grid_sample_frames)
            img = img.unsqueeze(1)
            im_array[i,:,:,:,:] = img #insert sweep to the first dimension
        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # im_list = [normalize(m) for m in torch.unbind(im_array, dim=0)]
        # img = torch.stack(im_list, dim=0)
        img = im_array
        return img, study_id

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def suppress_temporal_neighborhood(self, scores: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Greedy temporal suppression along the Nframe dimension.

        Args:
            scores: Tensor of shape [B, Ncines, Nframe, 1]
            radius: number of frames on each side to suppress

        Returns:
            Tensor of shape [B, Ncines, Nframe, 1] with nearby frames masked to -inf
            so that subsequent top-k won't pick temporally adjacent frames.
        """
        assert scores.ndim == 4, f"Expected [B, Ncines, Nframe, 1], got {scores.shape}"
        B, C, T, _ = scores.shape
        out = scores.clone()

        neg_inf = torch.finfo(scores.dtype).min

        for b in range(B):
            for c in range(C):
                # Flatten the last dim for processing
                cine_scores = out[b, c, :, 0]               # [T]
                mask = torch.ones(T, dtype=torch.bool, device=scores.device)
                order = torch.argsort(cine_scores, descending=True)

                for idx in order:
                    i = int(idx)
                    if mask[i]:
                        left  = max(0, i - radius)
                        right = min(T, i + radius + 1)
                        mask[left:right] = False
                        mask[i] = True   # keep current index active
                    else:
                        cine_scores[i] = neg_inf

                # write the modified cine back
                out[b, c, :, 0] = cine_scores

        return out
    
    def top_p_mask_scores(self, scores: torch.Tensor, p: float) -> torch.Tensor:
        """
        scores: [B, T], non-negative (e.g. probs in [0,1])
        returns: float mask in {0,1} with shape [B, T],
                where entries in the top-p nucleus (by *normalized* scores) are 1.
        """
        # detach to avoid gradients through discrete selection
        scores = scores.detach()

        # make sure scores are finite and non-negative
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        scores = torch.clamp(scores, min=0.0)

        # sort by score desc
        sorted_scores, sorted_idx = scores.sort(dim=-1, descending=True)  # [B, T]

        # normalize to sum to 1 per row (like probs)
        row_sum = sorted_scores.sum(dim=-1, keepdim=True)                 # [B, 1]
        norm_scores = sorted_scores / (row_sum + 1e-8)

        # cumulative normalized score
        cumprobs = norm_scores.cumsum(dim=-1)

        # keep indices until cumulative mass exceeds p
        keep = cumprobs <= p
        # always keep at least the top-3
        keep[..., :3] = True

        # map back to original order
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_idx, src=keep)

        return mask.float()   # [B, T] in {0,1}

    def forward(self, query, values):
        # ---------------------------------------------------------------------
        # 1. Compute raw logits 
        # ---------------------------------------------------------------------
        raw_before_suppression = self.V(torch.tanh(self.W1(query)))  # [..., 1] logits


        # ---------------------------------------------------------------------
        # 2. Apply suppression (for actual attention)
        # ---------------------------------------------------------------------
        radius = int(round(0.05 * raw_before_suppression.shape[-2]))
        raw = self.suppress_temporal_neighborhood(raw_before_suppression, radius)

        # Score in [0, 1]
        score = torch.sigmoid(raw)   # [B, Ncine, Nframe, 1]

        # ---------------------------------------------------------------------
        # 3. Flatten shapes
        # ---------------------------------------------------------------------
        B, Ncine, Nframe, _ = score.shape
        T = Ncine * Nframe

        flat_logits = raw.squeeze(-1).reshape(B, T)          # logits (for Gumbel)
        flat_probs  = score.squeeze(-1).reshape(B, T)        # scores in [0,1]

        # Sanitize probabilities
        flat_probs = torch.where(torch.isfinite(flat_probs),
                                flat_probs,
                                torch.zeros_like(flat_probs))

        # logits BEFORE suppression (for exploration only)
        raw_flat_logits = raw_before_suppression.squeeze(-1).reshape(B, T)
        raw_flat_logits_detached = raw_flat_logits.detach()

        # ---------------------------------------------------------------------
        # 4. Top-p selection (deterministic + Gumbel exploration)
        # ---------------------------------------------------------------------
        if self.training:
            top_p = 0.8        # nucleus threshold (tunable)
            eps   = 0.30       # exploration probability

            explore = (torch.rand(B, 1, device=flat_logits.device) < eps)

            # deterministic top-p on SUPPRESSED scores (exploitation)
            det_mask = self.top_p_mask_scores(flat_probs, top_p)   # [B, T]

            # stochastic "Gumbel" exploration on RAW logits → scores → top-p
            u = torch.rand_like(raw_flat_logits_detached).clamp_min(1e-12)
            g = -torch.log(-torch.log(u))
            logits_noisy = raw_flat_logits_detached + g            # [B, T]
            sto_scores   = torch.sigmoid(logits_noisy)             # [B, T] in (0,1)

            sto_scores = torch.where(torch.isfinite(sto_scores),
                                    sto_scores,
                                    torch.zeros_like(sto_scores))

            sto_mask = self.top_p_mask_scores(sto_scores, top_p)   # [B, T]

            # choose per batch: exploration vs exploitation
            # (broadcast explore from [B,1] to [B,T])
            mask = torch.where(explore, sto_mask, det_mask)

        else:
            top_p = 0.7        # eval nucleus threshold (tunable)
            mask = self.top_p_mask_scores(flat_probs, top_p)       # [B, T]

        # Make sure mask is finite and strictly 0/1
        mask = torch.where(torch.isfinite(mask), mask, torch.zeros_like(mask))
        mask = (mask > 0).float()

        # ---------------------------------------------------------------------
        # 5. Apply mask: keep selected entries (using *prob* values)
        # ---------------------------------------------------------------------
        flat_masked = flat_probs * mask                           # [B, T]

        masked_score = flat_masked.reshape(B, Ncine, Nframe, 1)
        score = masked_score

        # ---------------------------------------------------------------------
        # 6. Normalize to attention weights (safely)
        # ---------------------------------------------------------------------
        sum_score = torch.sum(score, (1, 2), keepdim=True)  # [B, 1, 1, 1]
        zero_mask = (sum_score <= 0)

        # base attention = score / (sum_score + eps)
        attention_weights = score / (sum_score + 1e-8)
        
        if zero_mask.any():
            # uniform weights over all frames for those problematic samples
            uniform = torch.full_like(
                attention_weights,
                1.0 / float(T), device=attention_weights.device)
            # Explicit shape expansion
            zero_mask_expanded = zero_mask.expand_as(attention_weights)
            attention_weights = torch.where(zero_mask_expanded, uniform, attention_weights)

        # ---------------------------------------------------------------------
        # 7. Context vector
        # ---------------------------------------------------------------------
        # values shape assumed [B, Ncine, Nframe, hidden_size]
        context_vector = attention_weights * values               # [B, Ncine, Nframe, H]
        context_vector = torch.sum(context_vector, dim=(1, 2))    # [B, H]

        return context_vector, attention_weights, score

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output


class GA_Net(nn.Module):
    def __init__(self):
        super(GA_Net, self).__init__()

        cnn = models.resnet50(pretrained=True)
        cnn.fc = Identity()

        self.TimeDistributed = TimeDistributed(cnn)
        self.WV = nn.Linear(2048, 128)
        self.Attention = SelfAttention(2048, 64)
        self.Prediction = nn.Linear(128, 1, bias=False)
        self.RMSNorm = nn.RMSNorm(2048)
        self.DropOut = nn.Dropout(p=0.1, inplace=True)
 
    def forward(self, x):

        if (len(x.shape) == 6):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4), x.size(5))
        else:
            multiple_cine_tags = False

        x = self.TimeDistributed(x)

        x = self.RMSNorm(x)
        x = self.DropOut(x)

        x_v = self.WV(x)

        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))
            x_v = x_v.view(batch_size, num_cine_tags, x_v.size(-2), x_v.size(-1))
        else:
            x = x.view(batch_size, 1, x.size(1), x.size(2))
            x_v = x_v.view(batch_size, 1, x_v.size(-2), x_v.size(-1))

        x_a, w_a, score = self.Attention(x, x_v)

        pred_by_frame = self.Prediction(x_v) * 100.0

        return torch.sum(w_a * pred_by_frame, dim=(1,2)), pred_by_frame, w_a, score

class GA_Net_features(nn.Module):
    def __init__(self, cnn_pretrained):
        super(GA_Net_features, self).__init__()

        cnn = cnn_pretrained

        self.TimeDistributed = TimeDistributed(cnn)

    def forward(self, x):
        
        if (len(x.shape) == 6):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4), x.size(5))
        else:
            multiple_cine_tags = False

        x = self.TimeDistributed(x)
        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))

        return x

class GA_Net_attn_output(nn.Module):
    def __init__(self):
        super(GA_Net_attn_output, self).__init__()

        self.WV = nn.Linear(2048, 128)
        self.Attention = SelfAttention(2048, 64)
        self.Prediction = nn.Linear(128, 1, bias=False)
        self.RMSNorm = nn.RMSNorm(2048)
        self.DropOut = nn.Dropout(p=0.1, inplace=True)
 
    def forward(self, x):

        if (len(x.shape) == 4):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3))
        else:
            multiple_cine_tags = False

        x = self.RMSNorm(x)
        x = self.DropOut(x)

        x_v = self.WV(x)
        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))
            x_v = x_v.view(batch_size, num_cine_tags, x_v.size(-2), x_v.size(-1))
        else:
            x = x.view(batch_size, 1, x.size(1), x.size(2))
            x_v = x_v.view(batch_size, 1, x_v.size(-2), x_v.size(-1))

        x_a, w_a, score = self.Attention(x, x_v)

        x = self.Prediction(x_a) * 100.0

        return x, score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv('/mnt/famli_netapp_shared/C1_ML_Analysis/famli_ml_lists/dataset_list/2025-11-04/dxa_instance_EFW_with_bw_dataset.csv')
test_df_ac = pd.read_csv('/mnt/famli_netapp_shared/C1_ML_Analysis/famli_ml_lists/dataset_list/2025-11-04/dxa_instance_EFW_AC_with_bw_dataset.csv')
test_df_ac['session_id'] = test_df_ac['session_id'] + '-flyto-ac'
test_df = pd.concat([test_df, test_df_ac], axis=0).reset_index(drop=True)
# test_df = test_df.loc[test_df.manufacturer == 'Butterfly Network Inc',:].reset_index(drop=True)
print('Number of visits: ', test_df.visitid.nunique())
# test_df = test_df[test_df["session_id"].str.contains("ga-tool-novice", na=False)]
test_df.file_path = test_df.file_path.str.replace('.*DXA/','DXA_masked_resampled_256_spc075/',regex=True)
test_df.file_path = test_df.file_path.str.replace('dcm','nrrd', regex=False)
test_df.file_path = test_df.file_path.str.replace('mp4','nrrd', regex=False)
test_df['efw_gt'] = test_df['birthweight_g']
test_df['study_id'] = test_df['session_id']
test_df['dataset'] = 'dxa'
sweep_counts = test_df.groupby("session_id").size().reset_index(name="count")
print('Session with less than 8 sweeps:')
print(sweep_counts[sweep_counts["count"] < 8])
test_df = test_df.loc[:,["study_id", "file_path", "efw_gt",'tag', 'dataset']].reset_index(drop=True)

# Check if each file exists
print('Number of unique session_id before excluding missing files: ', test_df.study_id.nunique())
test_df['file_exists'] = (dataset_dir + test_df['file_path']).apply(os.path.exists)
# assert test_df['file_exists'].all(), "Not all file paths exist!"
test_df = test_df.loc[test_df.file_exists,:].reset_index(drop=True)
print('Number of unique session_id after excluding missing files: ', test_df.study_id.nunique())

test_data = ITKImageDatasetByID(test_df, mount_point=dataset_dir,
                                ground_truth="efw_gt",
                                num_grid_sample_frames = 100,
                                cache=False, device='cpu')

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                          shuffle=False, drop_last=False, num_workers=4)

#Model
model = GA_Net()
model.load_state_dict(torch.load(model_path, map_location=device))

model_features = GA_Net_features(model.TimeDistributed.module)

model_attn_output = GA_Net_attn_output()
model_attn_output.WV = model.WV
model_attn_output.Attention = model.Attention
model_attn_output.Prediction = model.Prediction
model_attn_output.RMSNorm = model.RMSNorm
model_attn_output.DropOut = model.DropOut

model_features.to(device)
model_attn_output.to(device)

for param in model_features.parameters():
   param.requires_grad = False
for param in model_attn_output.parameters():
   param.requires_grad = False

# Loss
loss_fn = nn.MSELoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.to('cuda')

loss_fn2 = nn.L1Loss()
if torch.cuda.is_available():
    loss_fn2 = loss_fn2.to('cuda')


pred = []
attn_weights = []
study_ids = []
model_features.eval()
model_attn_output.eval()
mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float)[None, None, :, None, None].to(device)
sd = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float)[None, None, :, None, None].to(device)
with torch.no_grad():
    for batch, (X, study_id) in enumerate(test_dataloader):
        print(batch, study_id)
        batch_size = X.size(0)
        num_sweep = X.size(1)
        seq_len = X.size(2)
        features_list = []
        X = X.view(batch_size, num_sweep*seq_len, X.size(3), X.size(4), X.size(5))
        for x_chunk in torch.split(X, 10, dim=1):
            if torch.cuda.is_available():
                x_chunk = x_chunk.to(device, dtype=torch.float)
            x_chunk.div_(255)
            x_chunk = x_chunk.repeat_interleave(3,dim=2)
            x_chunk.sub_(mean).div_(sd)
            features_chunk = model_features(x_chunk)
            features_list.append(features_chunk)
        #Forward Pass
        features = torch.cat(features_list, dim=1)
        features = features.view(batch_size, num_sweep, seq_len, features.size(-1))
        out, scores = model_attn_output(features)
        out = out.view(batch_size)
        print(out.item())
        pred.append(out.cpu().numpy())
        attn_weights.append(scores.cpu().numpy())
        study_ids.append(study_id[0])


pred = np.hstack(pred)


test_results = pd.DataFrame(data = {"StudyID": study_ids,
                                    "pred": pred})

with open(os.path.join(output_dir, 'test_result_dxa_efw_bw_' + model_date_version + '.pickle'), 'wb') as f:
    pickle.dump(test_results, f)

with open(os.path.join(output_dir, 'test_result_dxa_efw_bw_' + model_date_version + '_attn_weights.pickle'), 'wb') as f:
    pickle.dump(attn_weights, f)
