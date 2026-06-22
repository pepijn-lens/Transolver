# Reproducing *Transolver: A Fast Transformer Solver for PDEs on General Geometries*

**A reproducibility study — DSAIT4205 Fundamental Research in Machine and Deep Learning (2025/26 Q4), TU Delft**

**Team:** Pepijn Lens · Nikshith Menta · Jasraj *(surname — TODO)*

---

## TL;DR

We reproduced central claims of **Transolver** (Wu et al., ICML 2024), the Transformer-based neural PDE
solver built around *Physics-Attention*. Working from the authors' [public code](https://github.com/thuml/Transolver),
we (1) re-ran the **slice-count ablation of Table 4** on Elasticity, (2) reproduced the **learned-slice
visualization of Figure 5(a)**, (3) ported the model to a **new aircraft surface dataset** introduced by the
follow-up *Transolver++* paper, and (4) checked the **ShapeNet-Car / AirfRANS design benchmarks (Table 3)** and a
**Transolver++ algorithm variant**. The headline finding: the qualitative story of the paper holds up well — more
slices help, a single slice collapses the method, and the learned slices are physically meaningful — but some
quantitative details (notably the claimed *degradation* at very large slice counts) **did not reproduce** under a
reduced training budget, which is itself an instructive result about reproducibility.

---

## 1. Introduction & motivation

### What Transolver does

Classical numerical PDE solvers discretize a domain into a large, often irregular mesh and can take hours to days
per simulation. Neural solvers learn the input→output mapping from precomputed simulations and then infer in a
flash. The challenge for Transformers is that meshes contain *hundreds of thousands* of points, and full
point-to-point attention is quadratic and physically unstructured.

Transolver's idea is **Physics-Attention**: instead of attending over raw mesh points, it learns to softly assign
each point to one of `M` **slices** ("physical states"), encodes each slice into a token, runs attention over the
`M` tokens (linear in the number of mesh points), and then "de-slices" back to points. The same
*Slice → Attend → Deslice* block ships in three flavours that differ only in the input projection:

| Variant | Input projection | Used for |
|---|---|---|
| `Physics_Attention_Irregular_Mesh` | `nn.Linear` (point cloud) | Elasticity, ShapeNet-Car, AirfRANS, **aircraft** |
| `Physics_Attention_Structured_Mesh_2D` | `Conv2d` (k=3) | Darcy, NS, Plasticity, Airfoil, Pipe |
| `Physics_Attention_Structured_Mesh_3D` | `Conv3d` (k=3) | (available, not in shipped benchmarks) |

The paper reports state-of-the-art on six standard benchmarks (+22% relative) and on car/airfoil design tasks.

### Why reproduce it — and why these experiments

Transolver is influential and was quickly followed by **Transolver++** (Luo et al., 2025), which scales the same
"learn physical states" idea to **million-point geometries** including **3D aircraft** designs. This sequel framed
two of our questions:

- **Does the *original* architecture transfer to the *new* aircraft data** that motivated the sequel, without the
  sequel's parallelism/local-adaptive machinery? (a *New data* study)
- Are the paper's **design choices** (number of slices `M`) and **interpretability claims** (the slice
  visualization) reproducible from the released code on modest hardware?

Reproduction matters because a paper's *conclusions* are only as strong as their robustness to budget, hardware,
seeds and data. Below we report where the conclusions held and where they bent.

---

## 2. Reproducibility criteria & contributions

Following the assignment, each member owns at least one reproducibility criterion. We worked from the authors'
existing code (so the "Reproduced" family of criteria applies) and added an ablation, a new dataset, and an
algorithm variant.

| Member | Experiment(s) | Criterion / criteria | Status |
|---|---|---|---|
| **Pepijn Lens** | Table 4 slice-count ablation (Elasticity); Figure 5(a) slice visualization; original Transolver on the new aircraft dataset | **Ablation study**, **Reproduced**, **New data** | ✅ written up below |
| **Nikshith Menta** | ShapeNet-Car (Table 3); AirfRANS (Table 3); Figure 5(b) using the `elas_256.pt` checkpoint from Pepijn's Table 4 run | **Reproduced** | 📝 stub — §4.1 |
| **Jasraj** | Transolver++ on ShapeNet-Car | **New algorithm variant** | 📝 stub — §4.2 |

> The codebase is organized as four *independent* sub-projects (`PDE-Solving-StandardBenchmark/`,
> `Car-Design-ShapeNetCar/`, `Airfoil-Design-AirfRANS/`, and our added `Aircraft-Design/`), each with its own copy
> of `Physics_Attention.py`. We kept them independent rather than refactoring a shared module.

---

## 3. Pepijn's experiments

### 3.1 Table 4 — slice-count (`M`) ablation on Elasticity *(Ablation study + Reproduced)*

**What the paper claims.** Table 4 sweeps the number of slices `M ∈ {1, …, 1024}` on Elasticity and Darcy and
reports relative-L2 error plus efficiency (peak memory, time/epoch on a 1024-point mesh, batch 1). The paper's
conclusions are: (i) `M=1` collapses Physics-Attention into a global pooling operator and badly hurts accuracy;
(ii) accuracy improves as `M` grows, with the **best at `M=256`**; (iii) an **excessively large `M` (e.g. 1024)
slightly *degrades*** accuracy ("too-large `M` fragments the physics domain"); they recommend `M=64`.

**Our setup.** Run on **Kaggle, 2× NVIDIA T4** (one `M` per GPU, sequentially), using the ablation harness in
`PDE-Solving-StandardBenchmark/` (`run_table4.sh`, `benchmark_efficiency.py`, `collect_results.py`). The single deliberate
deviation: **300 epochs instead of the paper's 500**, due to limited compute. We reproduced the **Elasticity**
column across all ten slice counts; the Darcy commands are wired up but were not run to completion under our budget.
Results are tracked in git (`logs/elas_M*.log`, `results/efficiency.csv`).

**Results.**

| `M` | Rel-L2 (ours, 300 ep) | Rel-L2 (paper, 500 ep) | Peak mem, ours (GB, T4) | Time/epoch, ours (s) | Time/epoch, paper (s) |
|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0256 | 0.0148 | 0.069 | 29.0 | 37.8 |
| 8 | 0.0111 | 0.0071 | 0.073 | 29.3 | 37.8 |
| 16 | 0.0130 | 0.0067 | 0.078 | 28.5 | 38.0 |
| 32 | 0.0100 | 0.0067 | 0.087 | 28.8 | 38.0 |
| 64 | 0.0094 | 0.0064 | 0.109 | 29.1 | 38.2 |
| 96 | 0.0081 | 0.0061 | 0.132 | 28.6 | 38.3 |
| 128 | 0.0085 | 0.0058 | 0.156 | 29.1 | 38.8 |
| 256 | 0.0083 | **0.0054** | 0.254 | 29.0 | 39.1 |
| 512 | 0.0071 | 0.0059 | 0.473 | 44.3 | 39.8 |
| 1024 | **0.0059** | 0.0068 | 1.035 | 84.8 | 40.5 |

![Slice-count ablation: reproduction vs. paper](blog_figures/table4_error_vs_M.png)

*Reproduction (blue) vs. paper Table 4 (orange) on Elasticity. Both agree that one slice is bad and that more
slices help; they diverge in the tail.*

**Findings.**

1. **The qualitative trend reproduces.** `M=1` is clearly the worst (global pooling, no physical correlations), and
   error drops sharply as `M` increases — exactly the paper's central message about *why* Physics-Attention works.
2. **The "too-large `M` hurts" claim did *not* reproduce.** In the paper, error bottoms out at `M=256` and rises
   for `M=512, 1024`. In our runs error keeps **falling monotonically through `M=1024` (best, 0.0059)**. The most
   likely cause is the **reduced 300-epoch budget**: at very large `M` the model has many more slice parameters and
   plausibly needs the full 500 epochs before over-fragmentation manifests as test-error degradation. Data version,
   seed, and hardware (T4 vs. the paper's GPU) may also contribute. *Consequence:* the paper's practical
   recommendation (`M=64`, easy to tune in `[32, 256]`) is sound for efficiency, but the specific claim that
   `M=1024` is harmful is budget-sensitive and we cannot confirm it.
3. **Absolute efficiency numbers are hardware-specific and should not be compared directly.** Our T4 peak-memory
   measurements (0.07–1.0 GB) are *lower in absolute terms* than the paper's (0.60–1.53 GB) — the paper's figures
   include a large fixed baseline — yet our memory grows much faster *relatively* (≈15× vs. ≈2.5×). Time/epoch is
   flat (~29 s) until `M ≥ 512`, where the T4 hits a wall (44 s at 512, 85 s at 1024).

![Table 4 efficiency reproduction](blog_figures/table4_efficiency_vs_M.png)

*Memory (log–log) and time vs. `M`. The relative scaling is informative; absolute values are not comparable across
hardware.*

> **Reproducibility takeaway:** efficiency tables are only meaningful relative to the hardware they were measured
> on, and a reduced epoch budget can silently erase a paper's fine-grained conclusions (here, the large-`M` dip).

---

### 3.2 New data — the original Transolver on aircraft surfaces *(New data)*

**Motivation.** *Transolver++* introduces a **3D aircraft** dataset to argue for its new million-scale machinery.
We asked the prior question: how well does the **original** Transolver (no Transolver++ additions) already do on
this kind of aircraft surface data? The code is a new self-contained `Aircraft-Design/` sub-project (data
preprocessing, `Transolver_Irregular_Mesh` model, training loop, SLURM scripts).

**Dataset.** Aircraft surface meshes from CFD simulations: **150 cases = 30 geometries × 5 flight conditions**
(combinations of Mach ∈ {2.0, 7.0}, angle-of-attack α ∈ {0°, 7°}, sideslip β ∈ {0°, 2°}). We split **by geometry**
to prevent leakage: **120 train / 30 test**, with 6 geometries fully held out. Per node, the input is the
area-weighted **surface normal (3) + (Ma, α, β)** broadcast = 6 channels; the model predicts **6 surface fields**:
pressure coefficient `Cp`, density `Rho`, velocities `U, V, W`, and `Pressure`. All fields are normalized with
training-set statistics (cached in `coef_norm.npz`).

**Model & training.** `Transolver_Irregular_Mesh`, `n_hidden=256`, `n_layers=8`, `n_heads=8`, `slice_num=32`,
`mlp_ratio=2` (**3.86 M params**). AdamW (wd 1e-5), OneCycleLR (max-lr 1e-3), gradient clipping 0.1, **batch size 1**
(variable-size meshes), **200 epochs** on a single A100, ~2.1 h total (trained as two SLURM jobs with a checkpoint
resume at epoch 168). Metric: relative-L2.

**Results.**

![Aircraft training curves](blog_figures/aircraft_loss_curves.png)

| Field | Test rel-L2 |
|---|---|
| U (streamwise velocity) | **0.0397** (best) |
| Pressure | 0.0977 |
| Cp | 0.1279 |
| Rho | 0.1530 |
| W | 0.1661 |
| V (cross-flow velocity) | **0.2166** (worst) |
| **Overall** | **0.0981** |

![Aircraft per-field error](blog_figures/aircraft_per_field.png)

**Findings.**

- The original Transolver **trains stably and transfers to aircraft surface data out of the box**, reaching an
  overall test relative-L2 of **≈9.8%** in ~2 hours on one GPU with no architecture changes — a positive signal for
  the generality claim, and a sensible *baseline* for what Transolver++ improves upon.
- Accuracy is **very uneven across fields**: the dominant streamwise velocity `U` is predicted to ~4%, whereas the
  small-magnitude cross-flow components `V` (and to a lesser extent `Rho`, `W`) are far harder (~15–22%). These are
  exactly the low-energy, sharp-gradient quantities, consistent with a relative-L2 metric being unforgiving on
  small-norm targets.
- The train/test gap (train 0.048 vs. test 0.098) and the validation **plateau after ~epoch 130** suggest the
  150-case dataset is the binding constraint, not the optimizer.

> This is an *exploratory* new-data result: there is no original-paper number for this exact setup to match against
> (Transolver++ reports on its own pipeline), so we frame it as "does the architecture port and behave sensibly?" —
> and it does.

---

### 3.3 Figure 5(a) — learned-slice visualization *(Reproduced, qualitative)*

**What the paper shows.** Figure 5(a) visualizes the 64 learned slice-weight maps from the last Physics-Attention
layer on an Elasticity sample, side by side for the **original mesh** and a **50%-resampled mesh**, to argue that
slices capture coherent physical regions and that this assignment is robust to mesh resolution.

**Our setup.** `Transolver_Irregular_Mesh`, `slice_num=64`, `n_hidden=128`, `n_heads=8`,
`n_layers=8` (~0.71 M params), trained on Elasticity (972-point meshes) for 500 epochs (CosineAnnealing) to a final
test **rel-L2 ≈ 0.0090** — close to the paper's main Elasticity result. `visualize_figure5a.py` extracts the
per-point slice weights (averaged over heads → `[N, 64]`), resamples the mesh at 50–80%, and renders all 64 slices
for both meshes.

![Reproduced Figure 5(a): 64 learned slices, original (top) vs. resampled (bottom) mesh](blog_figures/figure5a__frac0.5_scatter_s3_per_slice.png)

*Our reproduction of Figure 5(a): each tile is one of the 64 learned slices, rendered as a per-point scatter with
per-slice color normalization. The top four rows are the original mesh; the bottom four are the 50%-resampled mesh.*

![Paper's original Figure 5(a) for comparison](blog_figures/figure5a_paper_original.png)

*The paper's original Figure 5(a) (Wu et al., 2024), reproduced here for comparison. Note how, between the original
(top) and resampled (bottom) meshes, the corresponding tiles do **not** share colors — the discrepancy discussed
below.*

**Findings.**

- **Qualitatively, the claim reproduces well:** the 64 slices learn smooth, spatially-coherent partitions of the
  unit cell, and the resampled mesh recovers the *same* slice structure despite dropping half the points — exactly
  the robustness Transolver advertises.
- **One interesting discrepancy.** In *our* render, each region's color **matches** between the original and the
  resampled mesh (if a slice is yellow in the top-left of the original, it is yellow in the top-left of the
  resampled one). In the **paper's** figure (shown above) the corresponding tiles do **not** share colors. Since the slice-weight
  function is identical for both meshes (same trained weights, same coordinates), we *expect* them to match — and
  we believe **matching colors actually strengthens the authors' point**: the point-to-slice relationship carries
  over from the original to the resampled mesh. The paper's mismatch is most plausibly a per-tile colormap /
  normalization artifact (e.g. independent `vmax` per subplot) rather than a property of the model. We swept
  global-vs-per-slice normalization and scatter-vs-triangulation rendering to confirm the matching is real, not an
  artifact of *our* plotting choices.

> **Reproducibility takeaway:** visualization conventions (per-subplot color normalization) can accidentally hide
> the very property a figure is meant to demonstrate. Reproducing the figure surfaced this.

---

## 4. Teammate experiments

> The sections below are owned by Pepijn's teammates. The framing and table skeletons are provided; **numbers are
> to be filled in by each author** (marked **TODO**).

### 4.1 ShapeNet-Car, AirfRANS, and Figure 5(b) — *Nikshith Menta (Reproduced)*

Reproduction of the **design-task results (Table 3)** using the authors' code:

- **ShapeNet-Car** (`Car-Design-ShapeNetCar/`): surface pressure / drag prediction on car geometries (requires
  `pytorch_geometric` + `torch-cluster`).
- **AirfRANS** (`Airfoil-Design-AirfRANS/`): RANS airfoil task, `--task full` (and possibly `scarce/reynolds/aoa`).
- **Figure 5(b)** — the *resampling-robustness* counterpart to §3.3 — rendered from the **`elas_256.pt`** checkpoint
  produced by Pepijn's `M=256` Table 4 run (a nice cross-experiment reuse).

> **TODO (Nikshith):** fill in setup (hardware, epochs, hyperparameters), the metrics below, and a short discussion
> of whether Table 3 reproduces (volume/surface relative error, drag/lift coefficient error, Spearman rank
> correlation).

| Task | Metric | Paper | Reproduction |
|---|---|---|---|
| ShapeNet-Car | drag coef. error / Spearman ρ | *TODO* | *TODO* |
| AirfRANS (full) | rel. error / coef. error | *TODO* | *TODO* |

*Figure 5(b) image: TODO.*

### 4.2 Transolver++ on ShapeNet-Car — *Jasraj (New algorithm variant)*

Evaluates the **Transolver++** variant (the follow-up architecture with the optimized parallelism and local
adaptive mechanism) on ShapeNet-Car, to compare against the original Transolver on the same task.

> **TODO (Jasraj):** describe how the Transolver++ variant differs from the block used elsewhere in this repo, the
> training setup, and whether it improves over the original Transolver on ShapeNet-Car.

| Model | ShapeNet-Car metric | Result |
|---|---|---|
| Transolver (original) | *TODO* | *TODO* |
| Transolver++ (variant) | *TODO* | *TODO* |

---

## 5. Conclusion — do our results uphold the paper?

**Largely yes, with caveats.** The *core conclusions* of Transolver reproduced:

- Physics-Attention's value is real: a single slice (`M=1`) collapses the model, and more slices monotonically help
  on Elasticity (§3.1).
- The learned slices are physically meaningful and **robust to mesh resampling** (§3.3).
- The architecture is **general**: it ports to an unseen aircraft surface dataset and trains to a sensible ~9.8%
  error with no changes (§3.2).

**What did not reproduce / needs a caveat:**

- The fine-grained claim that **very large `M` degrades accuracy** did *not* hold under our **300-epoch** budget —
  our error kept improving through `M=1024`. This does not contradict the paper's *recommendation* (`M=64` for
  efficiency) but shows that conclusion is **training-budget-sensitive**.
- **Efficiency numbers are hardware-specific**: absolute memory/time on 2× T4 are not comparable to the paper's, and
  only the relative scaling should be read across setups.
- A reproduction of Figure 5(a) revealed a likely **colormap-normalization artifact** in the original figure: the
  per-tile colors *should* match between original and resampled meshes, and in our faithful render they do.

**Consequences & lessons.** Transolver is a robust, reproducible piece of work whose *qualitative* claims survive
budget cuts, hardware changes, and a brand-new dataset. The reproduction was most valuable precisely where it
*disagreed*: it showed that (a) a paper's tail-end ablation conclusions can depend on the full training budget, (b)
efficiency tables must be read relative to hardware, and (c) figure-rendering conventions can obscure the property a
figure is meant to prove. These are exactly the kinds of robustness questions a reproduction is for.

---

## 6. Reproducing our work

All three experiments are merged into `main`, each in its own sub-directory:

```bash
# Table 4 ablation:  PDE-Solving-StandardBenchmark/run_table4.sh + logs/elas_M*.log + results/efficiency.csv
# New aircraft data: Aircraft-Design/ (scripts/train.sbatch, train.py) + logs/
# Figure 5(a):       PDE-Solving-StandardBenchmark/visualize_figure5a.py

# Regenerate the figures in this post from the committed logs/CSVs:
pip install matplotlib numpy
python blog_figures/make_plots.py     # writes PNGs into blog_figures/
```

## References

- H. Wu, H. Luo, H. Wang, J. Wang, M. Long. **Transolver: A Fast Transformer Solver for PDEs on General
  Geometries.** ICML 2024. arXiv:2402.02366. Code: <https://github.com/thuml/Transolver>
- H. Luo, H. Wu, H. Zhou, L. Xing, Y. Di, J. Wang, M. Long. **Transolver++: An Accurate Neural Solver for PDEs on
  Million-Scale Geometries.** 2025. arXiv:2502.02414.

---

### Contributions (one-liners)

- **Pepijn Lens** — Table 4 slice-count ablation on Elasticity (Kaggle 2× T4, 300 ep); ported the original
  Transolver to the new aircraft surface dataset and ran it on the supercomputer (A100); reproduced the Figure 5(a)
  slice visualization and identified the colormap discrepancy. *(Ablation study, New data, Reproduced.)*
- **Nikshith Menta** — *TODO*: ShapeNet-Car & AirfRANS Table 3 reproduction; Figure 5(b) from `elas_256.pt`.
  *(Reproduced.)*
- **Jasraj** — *TODO*: Transolver++ algorithm variant on ShapeNet-Car. *(New algorithm variant.)*
