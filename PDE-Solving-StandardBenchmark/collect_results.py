"""
Parse training logs and efficiency CSV into a Table 4 summary.

Usage (from PDE-Solving-StandardBenchmark/):
    python collect_results.py

Expects:
    results/efficiency.csv   -- written by benchmark_efficiency.py
    logs/elas_M*.log         -- written by run_table4.sh
    logs/darcy_M*.log        -- written by run_table4.sh

Prints a table matching the paper's Table 4 layout.
"""

import re
import glob
import os
import csv

M_ORDER = [1, 8, 16, 32, 64, 96, 128, 256, 512, 1024]


def parse_final_rel_err(logfile):
    """Return the last rel_err value printed in a training log."""
    if not os.path.exists(logfile):
        return None
    text = open(logfile).read()
    matches = re.findall(r'rel_err\s*:\s*([\d.]+(?:e[+-]?\d+)?)', text)
    return float(matches[-1]) if matches else None


def load_efficiency(csv_path):
    """Return dict {M: (memory_gb, time_s)} from efficiency CSV."""
    result = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip() == 'M':
                continue  # skip blank lines and header
            try:
                m, mem, t = int(row[0]), float(row[1]), float(row[2])
                result[m] = (mem, t)
            except (ValueError, IndexError):
                pass
    return result


def fmt(val, decimals=4):
    if val is None:
        return '   N/A  '
    return f'{val:.{decimals}f}'


def main():
    eff = load_efficiency('results/efficiency.csv')

    header = (
        f"{'M':>6}  {'Memory(GB)':>10}  {'Time(s/ep)':>10}  "
        f"{'Elas L2':>9}  {'Darcy L2':>9}"
    )
    sep = '-' * len(header)
    print(sep)
    print(header)
    print(sep)

    for M in M_ORDER:
        mem_str = t_str = '       N/A'
        if M in eff:
            mem_str = f'{eff[M][0]:>10.4f}'
            t_str   = f'{eff[M][1]:>10.2f}'

        elas_l2 = parse_final_rel_err(f'logs/elas_M{M}.log')
        darcy_l2 = parse_final_rel_err(f'logs/darcy_M{M}.log')

        print(
            f'{M:>6}  {mem_str}  {t_str}  '
            f'{fmt(elas_l2):>9}  {fmt(darcy_l2):>9}'
        )

    # Regular Squares row (Darcy only)
    reg_l2 = parse_final_rel_err('logs/darcy_reg_squares.log')
    print(sep)
    print(
        f'{"Reg.Sq.":>6}  {"       N/A":>10}  {"       N/A":>10}  '
        f'{"       /":>9}  {fmt(reg_l2):>9}'
    )
    print(sep)

    print('\nNote: L2 values are from 300-epoch runs (paper uses 500).')
    print('Trends and rankings should match; absolute values may be ~5-10% higher.')


if __name__ == '__main__':
    main()
