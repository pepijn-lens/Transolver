"""Preprocess the raw aircraft Tecplot surface meshes into a fast .npz cache.

Raw layout (one Components.i.dat per case)::

    <data_dir>/<geom>/Mach<MM.MM>_Alpha<AA.AA>_Beta<BB.BB>/Components.i.dat

Each .dat is an ASCII Tecplot FEPOINT triangulation::

    TITLE = ...
    VARIABLES = x,y,z, Cp, Rho, U, V, W, Pressure
    ZONE T="Surface", N = <#nodes>, E=<#tris>, F=FEPOINT, ET=TRIANGLE
    <N lines: x y z Cp Rho U V W Pressure>
    <E lines: i j k>   (1-indexed triangle connectivity)

For every case we cache pos (N,3), per-node normals (N,3), the 6 target fields
(N,6) and the flight condition (Ma, alpha, beta). A train/test split (held out
by geometry, so no geometry leaks across the split) is written to
``airplane_dataset.json`` next to the cache.
"""
import os
import re
import json
import argparse
import numpy as np

FIELD_NAMES = ['Cp', 'Rho', 'U', 'V', 'W', 'Pressure']
COND_RE = re.compile(r'Mach([\d.]+)_Alpha([\d.]+)_Beta([\d.]+)')
ZONE_RE = re.compile(r'N\s*=\s*(\d+).*?E\s*=\s*(\d+)', re.IGNORECASE)


def parse_dat(path):
    """Return (pos[N,3], values[N,6], faces[E,3] 0-indexed)."""
    with open(path, 'r') as f:
        f.readline()  # TITLE
        f.readline()  # VARIABLES
        zone = f.readline()
        m = ZONE_RE.search(zone)
        if m is None:
            raise ValueError(f'Could not parse ZONE header in {path}: {zone!r}')
        n_nodes, n_elems = int(m.group(1)), int(m.group(2))
        nodes = np.loadtxt(f, max_rows=n_nodes, dtype=np.float64)
        faces = np.loadtxt(f, max_rows=n_elems, dtype=np.int64)
    assert nodes.shape == (n_nodes, 9), f'{path}: got node array {nodes.shape}'
    assert faces.shape == (n_elems, 3), f'{path}: got face array {faces.shape}'
    pos = nodes[:, 0:3]
    values = nodes[:, 3:9]
    faces = faces - 1  # Tecplot connectivity is 1-indexed
    return pos, values, faces


def compute_normals(pos, faces):
    """Area-weighted per-node normals from the triangulation, unit-normalized."""
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # magnitude == 2 * triangle area
    normals = np.zeros_like(pos)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return normals / norm


def cond_from_name(name):
    m = COND_RE.search(name)
    if m is None:
        raise ValueError(f'Could not parse flight condition from {name!r}')
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=np.float64)


def main():
    p = argparse.ArgumentParser('Preprocess aircraft .dat -> .npz cache')
    p.add_argument('--data_dir', type=str, default='../data/aircraft')
    p.add_argument('--save_dir', type=str, default='../data/aircraft_cache')
    p.add_argument('--n_test_geom', type=int, default=6,
                   help='# of geometries held out for the test split')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    geoms = sorted([d for d in os.listdir(args.data_dir)
                    if os.path.isdir(os.path.join(args.data_dir, d))])
    print(f'Found {len(geoms)} geometries under {args.data_dir}')

    all_cases = []  # (geom, npz_filename)
    for geom in geoms:
        gdir = os.path.join(args.data_dir, geom)
        for cond in sorted(os.listdir(gdir)):
            dat = os.path.join(gdir, cond, 'Components.i.dat')
            if not os.path.isfile(dat):
                continue
            out_name = f'{geom}__{cond}.npz'
            out_path = os.path.join(args.save_dir, out_name)
            all_cases.append((geom, out_name))
            if os.path.isfile(out_path):
                print(f'  skip (cached): {out_name}')
                continue
            print(f'  parsing {geom}/{cond} ...', flush=True)
            pos, values, faces = parse_dat(dat)
            normals = compute_normals(pos, faces)
            condition = cond_from_name(cond)
            np.savez_compressed(out_path,
                                pos=pos.astype(np.float32),
                                normals=normals.astype(np.float32),
                                values=values.astype(np.float32),
                                condition=condition.astype(np.float32),
                                field_names=np.array(FIELD_NAMES))

    # Split by geometry so no geometry appears in both train and test.
    rng = np.random.default_rng(args.seed)
    shuffled = list(geoms)
    rng.shuffle(shuffled)
    test_geoms = set(shuffled[:args.n_test_geom])
    train_set = [f for g, f in all_cases if g not in test_geoms]
    test_set = [f for g, f in all_cases if g in test_geoms]
    split = {'train_set': train_set, 'test_set': test_set,
             'test_geoms': sorted(test_geoms)}
    with open(os.path.join(args.save_dir, 'airplane_dataset.json'), 'w') as f:
        json.dump(split, f, indent=2)
    print(f'Wrote split: {len(train_set)} train / {len(test_set)} test cases '
          f'(test geoms: {sorted(test_geoms)})')


if __name__ == '__main__':
    main()
