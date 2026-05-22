"""Reproduce Figure 5(a): slice-weight visualization on Elasticity.

Loads a trained Transolver_Irregular_Mesh checkpoint, captures slice weights
from the last Physics-Attention layer (exposed as `last_slice_weights` on the
attention module), and renders the per-slice weight maps on both the original
mesh of a test sample and on a resampled (regular-grid) mesh covering the same
domain.
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Delaunay

from model_dict import get_model


def parse_args():
    p = argparse.ArgumentParser('Figure 5(a) visualization')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--checkpoint', type=str,
                   default='./checkpoints/elas_figure5a.pt')
    p.add_argument('--data_path', type=str,
                   default='/scratch/plens/FRML/data')
    p.add_argument('--sample_idx', type=int, default=0,
                   help='Test sample index used as the "original mesh".')
    p.add_argument('--slice_num', type=int, default=64)
    p.add_argument('--n_hidden', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--resample_grid', type=int, default=42,
                   help='Side length of the regular grid for the resampled '
                        'mesh; points outside the convex hull of the original '
                        'sample are dropped.')
    p.add_argument('--out_dir', type=str, default='./results_figure5a')
    p.add_argument('--ncols', type=int, default=16,
                   help='Columns per mesh in the figure grid.')
    return p.parse_args()


def build_model(args):
    class _A:
        pass
    margs = _A()
    margs.model = 'Transolver_Irregular_Mesh'
    model = get_model(margs).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=0.0,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        slice_num=args.slice_num,
        ref=8,
        unified_pos=0,
    ).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda'))
    model.eval()
    return model


def load_test_xy(data_path):
    path_xy = os.path.join(data_path, 'elasticity', 'Meshes',
                           'Random_UnitCell_XY_10.npy')
    arr = np.load(path_xy)
    arr = torch.tensor(arr, dtype=torch.float).permute(2, 0, 1)  # (N_total, N_pts, 2)
    return arr[-200:]  # exp_elas.py test split


def build_resampled_mesh(orig_xy, grid_side):
    xmin, ymin = orig_xy.min(0)
    xmax, ymax = orig_xy.max(0)
    gx = np.linspace(xmin, xmax, grid_side)
    gy = np.linspace(ymin, ymax, grid_side)
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    grid = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    # Keep only points inside the convex hull of the original mesh so the
    # resampled domain matches the unit-cell shape (with the hole).
    tri = Delaunay(orig_xy)
    inside = tri.find_simplex(grid) >= 0
    return grid[inside]


def run_and_capture(model, pts):
    with torch.no_grad():
        _ = model(pts, None)
    sw = model.blocks[-1].Attn.last_slice_weights  # [1, H, N, M]
    return sw[0].mean(dim=0).cpu().numpy()  # [N, M]


def plot_figure(orig_xy, orig_w, res_xy, res_w, ncols, out_dir):
    n_slices = orig_w.shape[1]
    nrows_per_mesh = int(np.ceil(n_slices / ncols))
    total_rows = 2 * nrows_per_mesh

    fig, axes = plt.subplots(total_rows, ncols,
                             figsize=(ncols * 0.9, total_rows * 0.9),
                             squeeze=False)

    # Per-slice vmin/vmax so weak slices remain visible.
    orig_vmax = orig_w.max(axis=0)
    res_vmax = res_w.max(axis=0)

    for s in range(n_slices):
        r = s // ncols
        c = s % ncols

        ax = axes[r, c]
        ax.scatter(orig_xy[:, 0], orig_xy[:, 1],
                   c=orig_w[:, s], s=1.5, cmap='coolwarm',
                   vmin=0.0, vmax=max(orig_vmax[s], 1e-6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax = axes[nrows_per_mesh + r, c]
        ax.scatter(res_xy[:, 0], res_xy[:, 1],
                   c=res_w[:, s], s=1.5, cmap='coolwarm',
                   vmin=0.0, vmax=max(res_vmax[s], 1e-6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Hide any leftover empty cells.
    for s in range(n_slices, nrows_per_mesh * ncols):
        r = s // ncols
        c = s % ncols
        axes[r, c].axis('off')
        axes[nrows_per_mesh + r, c].axis('off')

    fig.text(0.005, 0.75, 'Original\nMesh', ha='left', va='center',
             fontsize=10, rotation=0)
    fig.text(0.005, 0.25, 'Resampled\nMesh', ha='left', va='center',
             fontsize=10, rotation=0)
    fig.suptitle('(a) Learned Slice Visualization', y=0.02, fontsize=11)

    plt.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.04,
                        wspace=0.05, hspace=0.05)
    png_path = os.path.join(out_dir, 'figure5a.png')
    pdf_path = os.path.join(out_dir, 'figure5a.pdf')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args)
    test_xy = load_test_xy(args.data_path)

    orig_pts = test_xy[args.sample_idx:args.sample_idx + 1].cuda()  # [1, N, 2]
    orig_xy = orig_pts[0].cpu().numpy()

    res_xy = build_resampled_mesh(orig_xy, args.resample_grid)
    res_pts = torch.tensor(res_xy, dtype=torch.float).unsqueeze(0).cuda()

    orig_w = run_and_capture(model, orig_pts)
    res_w = run_and_capture(model, res_pts)

    np.save(os.path.join(args.out_dir, 'slice_weights_original.npy'), orig_w)
    np.save(os.path.join(args.out_dir, 'slice_weights_resampled.npy'), res_w)
    np.save(os.path.join(args.out_dir, 'mesh_original.npy'), orig_xy)
    np.save(os.path.join(args.out_dir, 'mesh_resampled.npy'), res_xy)

    png_path, pdf_path = plot_figure(orig_xy, orig_w, res_xy, res_w,
                                     args.ncols, args.out_dir)
    print(f'Original mesh:   {orig_xy.shape[0]} points')
    print(f'Resampled mesh:  {res_xy.shape[0]} points')
    print(f'Slice weights:   {orig_w.shape} (per-mesh, averaged over heads)')
    print(f'Saved figure to {png_path}')
    print(f'Saved figure to {pdf_path}')


if __name__ == '__main__':
    main()
