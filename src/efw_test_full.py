import argparse
import os

import torch

from loaders.ultrasound_dataset import USDatasetBlindSweepWTag
from loaders.transforms import ultrasound_transforms as ust
from torch.utils.data import DataLoader

from nets import efw
import pandas as pd

from lightning.pytorch.loggers import NeptuneLogger

from tqdm import tqdm

import pickle

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

    NN = getattr(efw, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    model = model.eval().cuda()

    df_test = pd.read_csv(args.csv)

    test_transform = ust.BlindSweepEvalTransforms()       

    test_ds = USDatasetBlindSweepWTag(df_test, mount_point=model.hparams.mount_point, img_column=model.hparams.img_column, tag_column=model.hparams.tag_column, ga_column=model.hparams.ga_column, id_column=model.hparams.id_column, frame_column=model.hparams.frame_column, class_column=model.hparams.class_column, presentation_column=model.hparams.presentation_column, efw_column=model.hparams.efw_column, max_sweeps=-1, transform=test_transform, num_frames=-1)

    dl = DataLoader(test_ds, batch_size=1, num_workers=8, prefetch_factor=2, persistent_workers=True, pin_memory=True, collate_fn=None)

    Y = []
    preds = []
    ids = []

    scores_chunks = []
    scores_frames = []
    tags_f = []

    preds_sweep = []
    ids_sweep = []
    scores_sweeps = []
    scores_chunks_sweep = []
    file_path = []
    Y_sweep = []

    for idx, batch in tqdm(enumerate(dl), total=len(dl)):        
        x = batch['img']  
        tags = batch['tag']        
        y = batch['efw']

        z = []
        z_t = []
        z_t_s = []

        
        Y.append(y.item())        
        ids.append(test_ds.keys[idx])
        group = test_ds.df_group.get_group(test_ds.keys[idx])
        tags_f.append(tags)

        with torch.no_grad():
            x = [x_.cuda(non_blocking=True) for x_ in x]
            tags = tags.cuda(non_blocking=True)

            for idx, x_sweep in enumerate(x):
                tag = tags[:, idx]
                
                x_sweep = x_sweep.permute(0, 2, 1, 3, 4)  # [BS, T, C, H, W]
                
                z_, z_t_, z_t_s_ = model.encode(x_sweep, tag)

                x_hat_, z_c_s_ = model.predict(z_t_)

                row = group.iloc[idx]
                preds_sweep.append(x_hat_.item())
                ids_sweep.append(row[model.hparams.id_column])
                scores_sweeps.append(z_t_s_.cpu())
                scores_chunks_sweep.append(z_c_s_.cpu())
                file_path.append(row[model.hparams.img_column])
                Y_sweep.append(y.item())

                z.append(z_)
                z_t.append(z_t_)
                z_t_s.append(z_t_s_)

            z = torch.cat(z, dim=1)  # [1, N_FRAMES, self.hparams.embed_dim]
            z_t = torch.cat(z_t, dim=1)  # [1, N_FRAMES, self.hparams.n_chunks, self.hparams.embed_dim]
            
            z_t_s = torch.cat(z_t_s, dim=1)  # [1, N_sweeps, T, self.hparams.n_chunks]

            z_t = z_t.view(1, -1, model.hparams.embed_dim)  # [BS, N_s*n_chunks, self.hparams.embed_dim]
            scores_frames.append(z_t_s.cpu())

            x_hat, z_c_s = model.predict(z_t)

            preds.append(x_hat.item())
            scores_chunks.append(z_c_s.cpu())

    Y = torch.tensor(Y)
    preds = torch.tensor(preds)

    abs_errors = torch.abs(Y - preds)
    mse = torch.mean((Y - preds) ** 2)
    mae = torch.mean(abs_errors)
    mape = torch.mean(abs_errors / Y) * 100.0    
    print(f'MSE: {mse.item():.2f}, MAE: {mae.item():.2f}, MAPE: {mape.item():.2f}')

    out_dir = os.path.join(args.out, args.model.split('/')[-2], os.path.splitext(os.path.basename(args.csv))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_dict = {
        'id': ids,
        'efw_gt': Y.tolist(),
        'efw_pred': preds.tolist(),
        'ids': ids,
        'tags': tags_f,
        'scores_frames': scores_frames,
        'scores_chunks': scores_chunks        
    }
    with open(os.path.join(out_dir, 'prediction.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)

    Y_sweep = torch.tensor(Y_sweep)
    preds_sweep = torch.tensor(preds_sweep)
    abs_errors = torch.abs(Y_sweep - preds_sweep)
    mse = torch.mean((Y_sweep - preds_sweep) ** 2)
    mae = torch.mean(abs_errors)
    mape = torch.mean(abs_errors / Y_sweep) * 100.0        
    print("Sweep-level results:")
    print(f'MSE: {mse.item():.2f}, MAE: {mae.item():.2f}, MAPE: {mape.item():.2f}')


    out_dict_sweep = {
        'id': ids_sweep,
        'efw_gt': Y_sweep,
        'efw_pred': preds_sweep,
        'scores_sweeps': scores_sweeps,
        'scores_chunks_sweep': scores_chunks_sweep,
        'file_path': file_path
    }
    with open(os.path.join(out_dir, 'prediction_sweep.pkl'), 'wb') as f:
        pickle.dump(out_dict_sweep, f)

    txt = "This test was performed using the following command:\n\n"
    txt += " ".join(os.sys.argv) + "\n\n"
    txt += f"Results:\nMSE: {mse.item():.2f}, MAE: {mae.item():.2f}, MAPE: {mape.item():.2f}\n"
    txt += f"Sweep-level results:\nMSE: {mse.item():.2f}, MAE: {mae.item():.2f}, MAPE: {mape.item():.2f}\n"
    txt += f"Number of samples: {len(Y)}\n"
    txt += f"Model checkpoint: {args.model}\n"

    with open(os.path.join(out_dir, 'test_info.txt'), 'w') as f:
        f.write(txt)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='EFW Test', add_help=False)

    input_group = parser.add_argument_group('Input')

    input_group.add_argument('--csv', help='CSV file path', type=str, default='/mnt/raid/C1_ML_Analysis/CSV_files/efw_2025-10-31_test.csv')
    input_group.add_argument('--nn', help='Type of neural network', type=str, required=True)
    input_group.add_argument('--model', help='Model for testing', type=str, required=True)
    input_group.add_argument('--out', help='Output dir', type=str, default='./test_output/fetal_biometry/efw/')


    input_group.add_argument('--neptune_tags', help='Neptune tags for logging', type=str, nargs='+', default=None)
    
    args = parser.parse_args()

    main(args)
