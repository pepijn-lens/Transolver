"""Dataset / loader for the cached aircraft surface meshes.

Reads the ``.npz`` cache produced by ``preprocess.py``. Each item is a full
surface mesh (one flight case); batch size is therefore effectively 1 and the
number of nodes varies between cases.

Inputs/outputs per item:
  - ``x``  : node coordinates,                ``(N, 3)``  (normalized)
  - ``fx`` : ``[normals(3), Ma, alpha, beta]``, ``(N, 6)`` (pos-independent geom + condition)
  - ``y``  : 6 surface fields (Cp,Rho,U,V,W,Pressure), ``(N, 6)`` (normalized)

Normalization coefficients (mean/std for pos, values, condition) are computed
once over the *training* split and shared with the test split.
"""
import os
import json
import numpy as np
import torch


def _stream_mean_std(save_dir, files, keys):
    """Per-channel mean/std for the given keys, streamed over all nodes."""
    counts = {k: 0 for k in keys}
    sums = {k: None for k in keys}
    sqs = {k: None for k in keys}
    for fn in files:
        d = np.load(os.path.join(save_dir, fn))
        for k in keys:
            a = np.atleast_2d(d[k]).astype(np.float64)
            if a.shape[0] == 1 and k == 'condition':
                pass  # condition is (3,) -> (1,3): one sample per case
            s = a.sum(axis=0)
            sq = (a ** 2).sum(axis=0)
            sums[k] = s if sums[k] is None else sums[k] + s
            sqs[k] = sq if sqs[k] is None else sqs[k] + sq
            counts[k] += a.shape[0]
    coef = {}
    for k in keys:
        mean = sums[k] / counts[k]
        var = np.maximum(sqs[k] / counts[k] - mean ** 2, 0.0)
        std = np.sqrt(var)
        std[std < 1e-8] = 1.0
        coef[k + '_mean'] = mean.astype(np.float32)
        coef[k + '_std'] = std.astype(np.float32)
    return coef


def compute_coef_norm(save_dir, train_files):
    return _stream_mean_std(save_dir, train_files, ['pos', 'values', 'condition'])


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir, split='train', coef_norm=None):
        self.save_dir = save_dir
        with open(os.path.join(save_dir, 'airplane_dataset.json'), 'r') as f:
            meta = json.load(f)
        self.files = meta['train_set'] if split == 'train' else meta['test_set']
        self.coef_norm = coef_norm

    def __len__(self):
        return len(self.files)

    def _norm(self, key, arr):
        if self.coef_norm is None:
            return arr
        m = self.coef_norm[key + '_mean']
        s = self.coef_norm[key + '_std']
        return (arr - m) / s

    def __getitem__(self, idx):
        d = np.load(os.path.join(self.save_dir, self.files[idx]))
        pos = self._norm('pos', d['pos'].astype(np.float32))          # (N,3)
        normals = d['normals'].astype(np.float32)                     # (N,3) unit vectors
        values = self._norm('values', d['values'].astype(np.float32))  # (N,6)
        cond = self._norm('condition', d['condition'].astype(np.float32))  # (3,)

        n = pos.shape[0]
        cond_b = np.broadcast_to(cond, (n, 3))
        fx = np.concatenate([normals, cond_b], axis=1)  # (N,6)

        x = torch.from_numpy(pos)
        fx = torch.from_numpy(np.ascontiguousarray(fx))
        y = torch.from_numpy(values)
        return x, fx, y
