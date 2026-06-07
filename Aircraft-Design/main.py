import os
import argparse
import numpy as np
import torch

from dataset.dataset import AircraftDataset, compute_coef_norm
from train import train_model, evaluate, FIELD_NAMES
from models.Transolver import Model


def get_args():
    p = argparse.ArgumentParser('Train Transolver on the aircraft surface task')
    p.add_argument('--save_dir', type=str, default='../data/aircraft_cache',
                   help='dir with the .npz cache + airplane_dataset.json (from preprocess.py)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--max_grad_norm', type=float, default=0.1)
    p.add_argument('--val_iter', type=int, default=10)
    # model
    p.add_argument('--n_hidden', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--mlp_ratio', type=int, default=2)
    p.add_argument('--slice_num', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--eval', type=int, default=0)
    p.add_argument('--save_name', type=str, default='aircraft_Transolver')
    return p.parse_args()


def main():
    args = get_args()
    print(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Normalization is fit on the training split only; cached for reuse at eval.
    coef_path = os.path.join(args.save_dir, 'coef_norm.npz')
    train_files_ds = AircraftDataset(args.save_dir, split='train', coef_norm=None)
    if os.path.isfile(coef_path):
        coef_norm = {k: v for k, v in np.load(coef_path).items()}
    else:
        print('Computing normalization coefficients over training split...', flush=True)
        coef_norm = compute_coef_norm(args.save_dir, train_files_ds.files)
        np.savez(coef_path, **coef_norm)
    print('values_mean:', coef_norm['values_mean'], 'values_std:', coef_norm['values_std'])

    train_ds = AircraftDataset(args.save_dir, split='train', coef_norm=coef_norm)
    val_ds = AircraftDataset(args.save_dir, split='test', coef_norm=coef_norm)

    model = Model(space_dim=3, fun_dim=6, out_dim=len(FIELD_NAMES),
                  n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_heads,
                  mlp_ratio=args.mlp_ratio, slice_num=args.slice_num,
                  dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Trainable Params: {n_params}')

    os.makedirs('./checkpoints', exist_ok=True)
    save_path = os.path.join('./checkpoints', args.save_name + '.pt')

    if args.eval:
        model.load_state_dict(torch.load(save_path, map_location=device))
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        val_err, per_field = evaluate(model, val_loader, device, coef_norm['values_std'])
        pf = '  '.join(f'{n}:{e:.4f}' for n, e in zip(FIELD_NAMES, per_field))
        print(f'[eval] val_relL2 {val_err:.5f}  [{pf}]')
    else:
        train_model(model, train_ds, val_ds, device, args, coef_norm, save_path)


if __name__ == '__main__':
    main()
