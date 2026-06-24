#!/usr/bin/env python3
"""Generate the figures embedded in ../BLOG.md from the committed experiment logs.

All numbers come from data tracked in this repository:
  * Aircraft (new-data) study  -> Aircraft-Design/logs/slurm-10121499.out (epochs 0-167)
                                  Aircraft-Design/logs/train.log         (epochs 168-199)
  * Table 4 slice ablation     -> committed on the `run-ablations` branch
                                  (logs/elas_M*.log + results/efficiency.csv).
                                  Those values are hard-coded below so this script is
                                  self-contained and runs from any branch.

Run:
    PYTHONPATH=/tmp/blogplot_libs python blog_figures/make_plots.py
(or with any interpreter that has matplotlib + numpy installed).

Outputs PNGs into the directory containing this script.
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AIR_LOGS = os.path.join(REPO, "Aircraft-Design", "logs")

# ---------------------------------------------------------------------------
# Aircraft training logs
# ---------------------------------------------------------------------------
# Line format (both files share it):
#   Epoch    0  train_relL2 0.71166  val_relL2 0.56618  [Cp:0.7387 Rho:... ]
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+train_relL2\s+([0-9.]+)"
    r"(?:\s+val_relL2\s+([0-9.]+)\s+\[(.*?)\])?"
)
FIELD_RE = re.compile(r"(\w+):([0-9.]+)")
FIELDS = ["Cp", "Rho", "U", "V", "W", "Pressure"]


def parse_aircraft_logs():
    """Return (epochs, train, val_epochs, val, last_field_dict)."""
    train_by_epoch = {}
    val_by_epoch = {}
    field_by_epoch = {}
    for fname in ("slurm-10121499.out", "slurm-10145084.out", "train.log"):
        path = os.path.join(AIR_LOGS, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                m = EPOCH_RE.search(line)
                if not m:
                    continue
                ep = int(m.group(1))
                train_by_epoch[ep] = float(m.group(2))
                if m.group(3) is not None:
                    val_by_epoch[ep] = float(m.group(3))
                    field_by_epoch[ep] = {
                        k: float(v) for k, v in FIELD_RE.findall(m.group(4))
                    }
    epochs = sorted(train_by_epoch)
    train = [train_by_epoch[e] for e in epochs]
    vep = sorted(val_by_epoch)
    val = [val_by_epoch[e] for e in vep]
    last_fields = field_by_epoch[max(field_by_epoch)] if field_by_epoch else {}
    return epochs, train, vep, val, last_fields


def plot_aircraft_curves():
    epochs, train, vep, val, _ = parse_aircraft_logs()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(epochs, train, color="#1f77b4", lw=1.8, label="train (relative L2)")
    ax.plot(vep, val, color="#d62728", lw=1.8, marker="o", ms=4,
            label="test/val (relative L2)")
    ax.axvline(167.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(168, ax.get_ylim()[1] * 0.92, " resume\n (epoch 168)",
            color="gray", fontsize=8, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Aircraft surface fields — Transolver training (200 epochs, A100)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "aircraft_loss_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out, f"(final train {train[-1]:.4f}, final val {val[-1]:.4f})")


def plot_aircraft_per_field():
    _, _, _, _, last_fields = parse_aircraft_logs()
    vals = [last_fields[f] for f in FIELDS]
    colors = ["#4c72b0" if v == min(vals) else
              "#c44e52" if v == max(vals) else "#55a868" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    bars = ax.bar(FIELDS, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("test relative L2 error")
    ax.set_title("Aircraft — final per-field test error (epoch 199)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "aircraft_per_field.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Table 4 slice-count ablation (run-ablations branch results + paper values)
# ---------------------------------------------------------------------------
M = np.array([1, 8, 16, 32, 64, 96, 128, 256, 512, 1024])
REPRO_ELAS = np.array([0.0256, 0.0111, 0.0130, 0.0100, 0.0094,
                       0.0081, 0.0085, 0.0083, 0.0071, 0.0059])
PAPER_ELAS = np.array([0.0148, 0.0071, 0.0067, 0.0067, 0.0064,
                       0.0061, 0.0058, 0.0054, 0.0059, 0.0068])
REPRO_MEM_GB = np.array([0.069, 0.073, 0.078, 0.087, 0.109,
                         0.132, 0.156, 0.254, 0.473, 1.035])
PAPER_MEM_GB = np.array([0.60, 0.60, 0.61, 0.62, 0.64,
                         0.68, 0.69, 0.81, 1.01, 1.53])
REPRO_TIME = np.array([29.0, 29.3, 28.5, 28.8, 29.1,
                       28.6, 29.1, 29.0, 44.3, 84.8])
PAPER_TIME = np.array([37.76, 37.82, 37.96, 38.00, 38.18,
                       38.31, 38.78, 39.13, 39.75, 40.49])


def plot_table4_error():
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(M, REPRO_ELAS, marker="o", color="#1f77b4",
            label="reproduction (2×T4, 300 ep)")
    ax.plot(M, PAPER_ELAS, marker="s", color="#ff7f0e", ls="--",
            label="paper (Table 4, 500 ep)")
    ax.axvline(256, color="#ff7f0e", ls=":", lw=1, alpha=0.6)
    ax.text(256, ax.get_ylim()[1] * 0.95, " paper best (M=256)",
            color="#ff7f0e", fontsize=8, va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(M)
    ax.set_xticklabels(M, rotation=45)
    ax.set_xlabel("number of slices  M")
    ax.set_ylabel("Elasticity relative L2 error")
    ax.set_title("Table 4 reproduction — slice-count ablation (Elasticity)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(HERE, "table4_error_vs_M.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_table4_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(M, REPRO_MEM_GB, marker="o", color="#1f77b4", label="reproduction (T4)")
    ax1.plot(M, PAPER_MEM_GB, marker="s", ls="--", color="#ff7f0e", label="paper")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(M); ax1.set_xticklabels(M, rotation=45)
    ax1.set_xlabel("number of slices  M")
    ax1.set_ylabel("peak GPU memory (GB)")
    ax1.set_title("Memory vs M")
    ax1.legend(); ax1.grid(alpha=0.3, which="both")

    ax2.plot(M, REPRO_TIME, marker="o", color="#1f77b4", label="reproduction (T4)")
    ax2.plot(M, PAPER_TIME, marker="s", ls="--", color="#ff7f0e", label="paper")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(M); ax2.set_xticklabels(M, rotation=45)
    ax2.set_xlabel("number of slices  M")
    ax2.set_ylabel("time per epoch (s)")
    ax2.set_title("Per-epoch time vs M")
    ax2.legend(); ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Table 4 reproduction — efficiency (1024-point mesh, batch 1)")
    fig.tight_layout()
    out = os.path.join(HERE, "table4_efficiency_vs_M.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Transolver+ on ShapeNet-Car (Jasraj, §4.2)
# ---------------------------------------------------------------------------

TPP_DIR = os.path.join(REPO, "Transolver_plus")
TPP_EPOCH_LOG = os.path.join(TPP_DIR, "output", "0", "200_0.5", "epoch_log.jsonl")
PAPER_SURF = 0.0745
PAPER_VOL = 0.0207


def parse_transolverplus_log():
    """Return (epochs, surf, vol) at the epochs where validation was run."""
    ep, surf, vol = [], [], []
    with open(TPP_EPOCH_LOG, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("surf_l2re") is not None:
                ep.append(int(d["epoch"]))
                surf.append(float(d["surf_l2re"]))
                vol.append(float(d["vol_l2re"]))
    return ep, surf, vol


def plot_transolverplus_car_curves():
    ep, surf, vol = parse_transolverplus_log()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ser, base, title, col in (
        (ax1, surf, PAPER_SURF, "Surf L2RE (surface pressure)", "#c44e52"),
        (ax2, vol, PAPER_VOL, "Volume L2RE (surrounding velocity)", "#55a868"),
    ):
        ax.plot(ep, ser, color=col, lw=1.8, marker="o", ms=4,
                label="Transolver+ (this work)")
        ax.axhline(base, color="gray", ls="--", lw=1.5,
                   label=f"Transolver baseline ({base:.4f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("relative L2")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Transolver+ on ShapeNet-Car — paper metrics vs epoch (200 epochs)")
    fig.tight_layout()
    out = os.path.join(HERE, "transolverplus_car_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out, f"(final surf {surf[-1]:.4f}, final vol {vol[-1]:.4f})")


def plot_transolverplus_car_ablation():
    """Bar chart of the slice/depth ablation runs vs the paper baselines."""
    runs = []
    for sub in ("car_Transolver_plus_L4_H256_S32", "car_Transolver_plus_L8_H256_S32"):
        p = os.path.join(TPP_DIR, "results", sub, "result.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
    # main_car.py run (best checkpoint) from the analysis summary
    summ = os.path.join(TPP_DIR, "output", "0", "200_0.5", "plots",
                        "analysis_summary.json")
    main = json.load(open(summ)) if os.path.exists(summ) else None

    labels, surf, vol = [], [], []
    if main is not None:
        labels.append("main\nL4·S32 (1.74M)")
        surf.append(main["best_surf_l2re"]); vol.append(main["best_vol_l2re"])
    for r in runs:
        labels.append(f"L{r['n_layers']}·S{r['slice_num']} ({r['nb_params']/1e6:.2f}M)")
        surf.append(r["surf_l2re"]); vol.append(r["vol_l2re"])

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w / 2, surf, w, color="#c44e52", label="Surf L2RE")
    ax.bar(x + w / 2, vol, w, color="#55a868", label="Volume L2RE")
    ax.axhline(PAPER_SURF, color="#c44e52", ls="--", lw=1.2,
               label=f"Transolver Surf baseline ({PAPER_SURF})")
    ax.axhline(PAPER_VOL, color="#55a868", ls="--", lw=1.2,
               label=f"Transolver Vol baseline ({PAPER_VOL})")
    for xi, (s, v) in enumerate(zip(surf, vol)):
        ax.text(xi - w / 2, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative L2 error")
    ax.set_title("Transolver+ ShapeNet-Car ablation vs. Transolver paper baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "transolverplus_car_ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    plot_aircraft_curves()
    plot_aircraft_per_field()
    plot_table4_error()
    plot_table4_efficiency()
    plot_transolverplus_car_curves()
    plot_transolverplus_car_ablation()
    # NOTE: the Figure 5(a) slice visualization
    # (blog_figures/figure5a__frac0.5_scatter_s3_per_slice.png) is produced by
    # PDE-Solving-StandardBenchmark/visualize_figure5a.py, not by this script.


if __name__ == "__main__":
    main()
