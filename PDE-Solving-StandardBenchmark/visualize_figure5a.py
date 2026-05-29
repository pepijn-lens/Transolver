"""Reproduce Figure 5(a): slice-weight visualization on Elasticity.

Loads a trained Transolver_Irregular_Mesh checkpoint, captures slice weights
from the last Physics-Attention layer (exposed as ``last_slice_weights`` on
the attention module), and renders the per-slice weight maps on both the
original mesh and a resampled mesh (random subset of the original points,
50-80%, matching the paper's appendix description).

To make it easy to compare visual choices, this script sweeps over plotting
styles (scatter at several marker sizes vs. tripcolor), vmax normalization
(global vs. per-slice), and resample fractions, saving each variant to its
own file in --out_dir.
"""

import os
import argparse
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from model_dict import get_model


# ---------------------------------------------------------------------------
# Sweep configuration (edit here to add/remove variants).
# ---------------------------------------------------------------------------
RESAMPLE_FRACS = (0.5, 0.6, 0.7, 0.8)
SCATTER_MARKER_SIZES = (1.5, 3, 6)
VMAX_MODES = ('global', 'per_slice')
CANONICAL_VARIANT = dict(frac=0.5, style='tripcolor',
                         marker_size=0, vmax_mode='global')


def parse_args():
    p = argparse.ArgumentParser('Figure 5(a) visualization')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--checkpoint', type=str,
                   default='./checkpoints/elas_figure5a.pt')
    p.add_argument('--data_path', type=str,
                   default='/scratch/plens/FRML/data')
    p.add_argument('--sample_indices', type=str, default='0',
                   help='Comma-separated list of test sample indices to '
                        'visualize. Each gets its own out_dir/sample_<idx>/ '
                        'subfolder.')
    p.add_argument('--slice_num', type=int, default=64)
    p.add_argument('--n_hidden', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--out_dir', type=str, default='./results_figure5a')
    p.add_argument('--ncols', type=int, default=16,
                   help='Columns per mesh in the figure grid.')
    p.add_argument('--cmap', type=str, default='viridis')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for the random subsampling of the '
                        'resampled mesh.')
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
    arr = torch.tensor(arr, dtype=torch.float).permute(2, 0, 1)
    return arr[-200:]  # exp_elas.py test split


def random_subsample(xy, frac, rng):
    """Keep `frac` of the rows (paper's resampling protocol: 50-80%)."""
    n = xy.shape[0]
    keep = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    return xy[idx]


def run_and_capture(model, pts):
    with torch.no_grad():
        _ = model(pts, None)
    sw = model.blocks[-1].Attn.last_slice_weights  # [1, H, N, M]
    return sw[0].mean(dim=0).cpu().numpy()  # [N, M]


def _build_masked_triangulation(xy):
    """Delaunay triangulation, with long triangles (those bridging the
    central hole or hanging off the convex hull) masked out."""
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    pts = xy[tri.triangles]                    # (T, 3, 2)
    edges = np.diff(np.concatenate([pts, pts[:, :1]], axis=1), axis=1)
    edge_lens = np.linalg.norm(edges, axis=-1)  # (T, 3)
    max_edge = edge_lens.max(axis=1)
    threshold = 3.0 * np.median(max_edge)
    tri.set_mask(max_edge > threshold)
    return tri


def _draw_panel(ax, xy, w, *, style, marker_size, cmap, vmin, vmax, tri=None):
    if style == 'scatter':
        ax.scatter(xy[:, 0], xy[:, 1], c=w, s=marker_size, cmap=cmap,
                   vmin=vmin, vmax=vmax, marker='s', linewidths=0)
    elif style == 'tripcolor':
        if tri is None:
            tri = _build_masked_triangulation(xy)
        ax.tripcolor(tri, w, cmap=cmap, vmin=vmin, vmax=vmax,
                     shading='gouraud')
    else:
        raise ValueError(style)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_figure(orig_xy, orig_w, res_xy, res_w, *, out_path,
                ncols, style, marker_size, vmax_mode, cmap):
    n_slices = orig_w.shape[1]
    nrows_per_mesh = int(np.ceil(n_slices / ncols))
    total_rows = 2 * nrows_per_mesh

    fig, axes = plt.subplots(total_rows, ncols,
                             figsize=(ncols * 0.9, total_rows * 0.9),
                             squeeze=False)

    if vmax_mode == 'global':
        gmin = float(min(orig_w.min(), res_w.min()))
        gmax = float(max(orig_w.max(), res_w.max()))
    elif vmax_mode == 'per_slice':
        gmin = None
        gmax = None
    else:
        raise ValueError(vmax_mode)

    orig_tri = _build_masked_triangulation(orig_xy) if style == 'tripcolor' else None
    res_tri = _build_masked_triangulation(res_xy) if style == 'tripcolor' else None

    for s in range(n_slices):
        r = s // ncols
        c = s % ncols

        if vmax_mode == 'per_slice':
            vmin_o = float(orig_w[:, s].min())
            vmax_o = float(orig_w[:, s].max())
            vmin_r = float(res_w[:, s].min())
            vmax_r = float(res_w[:, s].max())
        else:
            vmin_o = vmin_r = gmin
            vmax_o = vmax_r = gmax

        _draw_panel(axes[r, c], orig_xy, orig_w[:, s],
                    style=style, marker_size=marker_size,
                    cmap=cmap, vmin=vmin_o, vmax=vmax_o, tri=orig_tri)
        _draw_panel(axes[nrows_per_mesh + r, c], res_xy, res_w[:, s],
                    style=style, marker_size=marker_size,
                    cmap=cmap, vmin=vmin_r, vmax=vmax_r, tri=res_tri)

    for s in range(n_slices, nrows_per_mesh * ncols):
        r = s // ncols
        c = s % ncols
        axes[r, c].axis('off')
        axes[nrows_per_mesh + r, c].axis('off')

    fig.text(0.012, 0.75, 'Original Mesh', ha='center', va='center',
             rotation=90, fontsize=11)
    fig.text(0.012, 0.27, 'Resampled Mesh', ha='center', va='center',
             rotation=90, fontsize=11)
    fig.suptitle('(a) Learned Slice Visualization', y=0.02, fontsize=12)

    plt.subplots_adjust(left=0.035, right=0.995, top=0.97, bottom=0.04,
                        wspace=0.03, hspace=0.03)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def variant_tag(frac, style, marker_size, vmax_mode):
    if style == 'scatter':
        return f'frac{frac:g}_scatter_s{marker_size:g}_{vmax_mode}'
    return f'frac{frac:g}_{style}_{vmax_mode}'


def render_sample(model, test_xy, sample_idx, args, sample_out_dir):
    os.makedirs(sample_out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed + sample_idx)

    orig_pts = test_xy[sample_idx:sample_idx + 1].cuda()
    orig_xy = orig_pts[0].cpu().numpy()
    orig_w = run_and_capture(model, orig_pts)
    np.save(os.path.join(sample_out_dir, 'mesh_original.npy'), orig_xy)
    np.save(os.path.join(sample_out_dir, 'slice_weights_original.npy'), orig_w)

    print(f'[sample {sample_idx}] {orig_xy.shape[0]} points')

    n_saved = 0
    for frac in RESAMPLE_FRACS:
        res_xy = random_subsample(orig_xy, frac, rng)
        res_pts = torch.tensor(res_xy, dtype=torch.float).unsqueeze(0).cuda()
        res_w = run_and_capture(model, res_pts)
        np.save(os.path.join(sample_out_dir,
                             f'mesh_resampled_frac{frac:g}.npy'), res_xy)
        np.save(os.path.join(sample_out_dir,
                             f'slice_weights_resampled_frac{frac:g}.npy'),
                res_w)
        print(f'  frac={frac:g}: kept {res_xy.shape[0]} / {orig_xy.shape[0]}')

        styles = (
            [('scatter', m) for m in SCATTER_MARKER_SIZES]
            + [('tripcolor', 0)]
        )
        for (style, m), vmax_mode in product(styles, VMAX_MODES):
            tag = variant_tag(frac, style, m, vmax_mode)
            out_path = os.path.join(sample_out_dir, f'figure5a__{tag}.png')
            plot_figure(orig_xy, orig_w, res_xy, res_w,
                        out_path=out_path,
                        ncols=args.ncols, style=style, marker_size=m,
                        vmax_mode=vmax_mode, cmap=args.cmap)
            n_saved += 1

    # Canonical figure for this sample.
    c = CANONICAL_VARIANT
    res_xy = random_subsample(orig_xy, c['frac'],
                              np.random.default_rng(args.seed + sample_idx))
    res_pts = torch.tensor(res_xy, dtype=torch.float).unsqueeze(0).cuda()
    res_w = run_and_capture(model, res_pts)
    for ext in ('png', 'pdf'):
        out_path = os.path.join(sample_out_dir, f'figure5a.{ext}')
        plot_figure(orig_xy, orig_w, res_xy, res_w, out_path=out_path,
                    ncols=args.ncols, style=c['style'],
                    marker_size=c['marker_size'],
                    vmax_mode=c['vmax_mode'], cmap=args.cmap)
    print(f'  saved {n_saved} variants + canonical to {sample_out_dir}/')
    return n_saved


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    sample_indices = [int(s) for s in args.sample_indices.split(',') if s.strip()]
    if not sample_indices:
        raise ValueError('--sample_indices must contain at least one index')

    model = build_model(args)
    test_xy = load_test_xy(args.data_path)

    total = 0
    for sidx in sample_indices:
        if sidx < 0 or sidx >= test_xy.shape[0]:
            raise IndexError(f'sample_idx {sidx} out of range '
                             f'[0, {test_xy.shape[0]})')
        sample_dir = os.path.join(args.out_dir, f'sample_{sidx:03d}')
        total += render_sample(model, test_xy, sidx, args, sample_dir)

    print(f'Done. {total} variant figures across {len(sample_indices)} '
          f'samples (plus canonicals).')


if __name__ == '__main__':
    main()
