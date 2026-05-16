"""
Efficiency benchmark for Table 4 of Transolver (ICML 2024).

Reproduces the #Memory and #Time columns: runs 1000 forward+backward passes on
a synthetic (1, 1024, 2) input with batch_size=1, matching the paper's footnote
"Efficiency is calculated on inputs with 1024 unstructured mesh points and batch size as 1."

Usage:
    python benchmark_efficiency.py --gpu 0 --slice_num 64
Output (CSV row, append to results/efficiency.csv):
    M,memory_gb,time_s_per_epoch
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.Transolver_Irregular_Mesh import Model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--slice_num', type=int, default=64)
parser.add_argument('--n_warmup', type=int, default=20,
                    help='warmup steps before timing')
parser.add_argument('--n_steps', type=int, default=1000,
                    help='measured steps (paper epoch size = 1000 samples, batch=1)')
parser.add_argument('--header', action='store_true',
                    help='print CSV header then exit')
args = parser.parse_args()

if args.header:
    print('M,memory_gb,time_s_per_epoch')
    sys.exit(0)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda')

N, B = 1024, 1

# Reset peak memory BEFORE model creation so we capture model weights +
# optimizer state (m, v buffers) + activations — matching the paper's
# "total GPU memory during training" measurement.
torch.cuda.reset_peak_memory_stats(device)

x = torch.randn(B, N, 2, device=device)
dummy_y = torch.zeros(B, N, device=device)

model = Model(
    space_dim=2,
    n_layers=8,
    n_hidden=128,
    dropout=0.0,
    n_head=8,
    Time_Input=False,
    mlp_ratio=1,
    fun_dim=0,
    out_dim=1,
    slice_num=args.slice_num,
    unified_pos=False,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Warmup (optimizer state buffers are allocated on the first step)
model.train()
for _ in range(args.n_warmup):
    optimizer.zero_grad()
    out = model(x, None).squeeze(-1)
    loss_fn(out, dummy_y).backward()
    optimizer.step()

# Benchmark — do NOT reset peak stats here; we want to include everything above
torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(args.n_steps):
    optimizer.zero_grad()
    out = model(x, None).squeeze(-1)
    loss_fn(out, dummy_y).backward()
    optimizer.step()

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
time_per_epoch = elapsed  # args.n_steps steps = 1 simulated epoch

print(f'{args.slice_num},{mem_gb:.4f},{time_per_epoch:.4f}')
