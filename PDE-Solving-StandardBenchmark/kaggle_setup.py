"""
Run this script as the first cell in your Kaggle notebook.

It installs dependencies, auto-detects data paths from the known Kaggle dataset
structure, exports ELAS_PATH / DARCY_PATH as environment variables, and runs a
2-epoch sanity check to confirm everything works before you commit GPU time.

Usage inside a Kaggle notebook cell:
    !python /kaggle/working/transolver/PDE-Solving-StandardBenchmark/kaggle_setup.py
"""

import os
import subprocess
import sys

REPO = '/kaggle/working/transolver/PDE-Solving-StandardBenchmark'
DATA_ROOT = '/kaggle/input/datasets/thegreenier/transolver-data-group9'


# ── 1. Install dependencies ──────────────────────────────────────────────────
print('=== Installing dependencies ===')
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-r',
     os.path.join(REPO, 'requirements.txt')],
    check=True,
)


# ── 2. Locate data files ─────────────────────────────────────────────────────
def find_file(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return dirpath
    return None


print('\n=== Locating datasets ===')

# Elasticity: needs Random_UnitCell_sigma_10.npy one level above elasticity/Meshes/
elas_meshes = find_file(DATA_ROOT, 'Random_UnitCell_sigma_10.npy')
if elas_meshes:
    # elas_meshes is the Meshes/ dir; code expects --data_path to be two levels up
    elas_path = os.path.dirname(os.path.dirname(elas_meshes))
    print(f'  Elasticity data_path: {elas_path}')
else:
    print('  ERROR: Random_UnitCell_sigma_10.npy not found under', DATA_ROOT)
    elas_path = None

# Darcy: needs piececonst_r421_N1024_smooth1.mat
darcy_dir = find_file(DATA_ROOT, 'piececonst_r421_N1024_smooth1.mat')
if darcy_dir:
    print(f'  Darcy data_path: {darcy_dir}')
else:
    print('  ERROR: piececonst_r421_N1024_smooth1.mat not found under', DATA_ROOT)
    darcy_dir = None

# Write paths to a shell-sourceable file so run_table4.sh can pick them up
env_file = os.path.join(REPO, 'table4_paths.env')
with open(env_file, 'w') as f:
    f.write(f'export ELAS_PATH="{elas_path or ""}"\n')
    f.write(f'export DARCY_PATH="{darcy_dir or ""}"\n')
print(f'\nPaths written to {env_file}')
print('Source before running scripts:  source table4_paths.env')


# ── 3. Sanity check (2 epochs, Elasticity M=64) ──────────────────────────────
if elas_path:
    print('\n=== Sanity check: 2 epochs of Elasticity M=64 ===')
    result = subprocess.run(
        [
            sys.executable, 'exp_elas.py',
            '--gpu', '0',
            '--model', 'Transolver_Irregular_Mesh',
            '--n-hidden', '128', '--n-heads', '8', '--n-layers', '8',
            '--lr', '0.001', '--max_grad_norm', '0.1',
            '--batch-size', '1', '--slice_num', '64', '--epochs', '2',
            '--unified_pos', '0',
            '--data_path', elas_path,
            '--save_name', 'sanity_check',
        ],
        cwd=REPO,
        capture_output=False,
    )
    if result.returncode == 0:
        print('\n✓ Sanity check passed. Ready to run experiments.')
    else:
        print('\n✗ Sanity check failed — check the output above.')
else:
    print('\nSkipping sanity check (Elasticity path not found).')
