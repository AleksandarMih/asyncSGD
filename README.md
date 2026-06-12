# Asynchronous SGD — Optimization for ML (CS-439)

An empirical study of how **gradient delay** affects the convergence of SGD on ResNet-20 / CIFAR-10, and how it interacts with momentum, learning-rate scaling, and classical regularizers (dropout, L2). Experiments cover deterministic fixed-lag delays, geometric (stochastic) delays that simulate a pool of asynchronous workers, and true Hogwild!-style async SGD.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Repository Structure](#repository-structure)
  - [src/](#src)
  - [scripts/](#scripts)
  - [scripts/momentum\_sweeps/](#scriptsmomentumsweeps)
  - [scripts/regularization\_sweeps/](#scriptsregularizationsweeps)
  - [outputs/](#outputs)
- [Root-Level Files](#root-level-files)
- [License](#license)
- [Authors](#authors)

---

## Environment Setup

Create and activate the Conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate OptML
```

---

## Repository Structure

### src/

Core model and data-loading code shared by all experiments.

| File | Description |
|---|---|
| `model.py` | ResNet-20 implementation for CIFAR-10 (0.27 M parameters). Includes `BasicBlock` with optional dropout, He initialization, and Option-A shortcuts. Also exposes `resnet32`, `resnet44`, `resnet56`, `resnet110`, and `resnet1202` factories. Architecture adapted from [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) (Yerlan Idelbayev). |
| `data.py` | CIFAR-10 data loading. `get_loaders()` returns train (batch 128) and test (batch 256) `DataLoader`s with standard augmentation (random crop, horizontal flip) and channel-wise normalization. |

---

### scripts/

Main training script, per-experiment analysis scripts, and figure generation.

| File | Description |
|---|---|
| `train.py` | Central training script. Implements a custom `SGDMomentum` optimizer (supports negative β), a FIFO delay queue for fixed-lag experiments, a geometric delay queue that simulates M async workers (τ ~ Geom(1/M)−1), and a Hogwild!-style async training loop. Exposes helpers for implicit-momentum estimation, momentum compensation, Liu-bound computation, and adaptive momentum schedules. All experiments are launched by calling this script with appropriate CLI flags. |
| `analyze_det_grid.py` | Loads deterministic-delay grid results and produces three plots: final test accuracy vs. τ per β value, training-loss curves at τ=10, and the momentum-benefit headline (accuracy gain of β=0.9 over β=0 as a function of τ). |
| `analyze_det_theory.py` | Loads Step-3 baselines plus Step-4 theory runs and produces five plots covering the equivalence test (Exp A), momentum compensation (Exp B), LR-scaling rules (Exp C), gradient alignment over training, and convergence-speed comparison across experiments. |
| `analyze_geo.py` | Compares deterministic and geometric delay; produces accuracy-vs-E[τ] comparison plots, the geometric-delay equivalence test, and an adaptive-schedule comparison between delay types. |
| `analyze_step5_ext.py` | Analyses Ilia's Step-5 Hogwild! results combined with the LR-rescue extension; produces final-accuracy grids and LR-scaling rescue efficiency plots. |
| `make_figures.py` | Generates publication-quality PDF + PNG figures (300 dpi, DejaVu Serif, precise column widths) by parsing result filenames with regex and aggregating CSV metrics. |
| `compute_q1_metrics.py` | Aggregates deterministic-grid results and reports convergence metrics: epochs to 80 % and 85 % test accuracy per τ, and normalized area under the convergence curve. |

---

### scripts/momentum\_sweeps/

Shell scripts that launch the main momentum-and-delay experiments by calling `scripts/train.py` in batch.

| File | Description |
|---|---|
| `sweep_det_grid.sh` | Step 3 baseline: sweeps a 5 τ × 3 β × 3 seed grid with deterministic (FIFO) delays to characterize how delay and momentum interact. Results written to `outputs/det_grid/`. |
| `sweep_det_theory.sh` | Step 4 theory experiments A–C: tests the implicit-momentum equivalence hypothesis (Exp A), explicit-momentum compensation (Exp B), and LR-scaling rules 1/τ and 1/√τ (Exp C). Results written to `outputs/det_theory/`. |
| `sweep_adaptive_det.sh` | Exp G: sweeps adaptive momentum schedules that track the Liu stability bound over training, for deterministic delays. |
| `sweep_above_bound.sh` | Exp H: probes what happens when the explicit momentum is set above the Liu stability bound. |
| `sweep_geo_grid.sh` | Geometric-delay equivalent of `sweep_det_grid.sh`; uses a stochastic geometric queue parameterized by M workers. Results written to `outputs/geo_grid/`. |
| `sweep_geo_large.sh` | Large-scale geometric-delay sweeps covering a wider (τ, β) range. Results written to `outputs/geo_large/`. |
| `sweep_geo_equivalence.sh` | Runs the Mitliagkas equivalence test on geometric delays to check whether E[τ] and deterministic τ produce the same implicit-momentum effect. |
| `sweep_neg_momentum.sh` | Tests negative β values as a compensation mechanism for large gradient delays. |
| `sweep_neg_momentum_ext.sh` | Extended negative-momentum sweep over a broader β range. |
| `sweep_lrs_rescue.sh` | LR-scaling rescue experiments for deterministic delays: measures whether a 1/τ or 1/√τ LR rule recovers accuracy lost to large delays. |
| `sweep_lrs_rescue_geo.sh` | Same LR-scaling rescue experiment applied to geometric delays. |
| `sweep_ilia_beta0.sh` | β=0 (pure SGD) baseline sweeps used as the reference point for Step-5 async experiments. |
| `sweep_step5.sbatch` | SLURM batch-submission script for running Step-5 sweeps on a cluster. |
| `make_configs_step5.sh` | Generates the hyperparameter-config files consumed by `sweep_step5.sbatch`. |

---

### scripts/regularization\_sweeps/

Python scripts that investigate how gradient delay interacts with classical regularization techniques.

| File | Description |
|---|---|
| `sweep_tau_beta.py` | Sweeps the (τ, β) grid with a geometric delay queue to characterize the delay–momentum interaction as a potential regularization effect. |
| `sweep_dropout.py` | Sweeps dropout rates [0.0, 0.05, …, 0.4] at zero delay to establish a dropout regularization baseline for comparison with delay. |
| `sweep_l2.py` | Sweeps L2 weight-decay values at zero delay to establish an L2 regularization baseline. |
| `sweep_tau_dropout.py` | Cross-sweep of delay τ and dropout rate to decompose and compare their individual and combined regularization effects. |
| `sweep_tau_l2.py` | Cross-sweep of delay τ and weight decay to decompose and compare their individual and combined effects. |
| `sweep_grad_bound.py` | Tests gradient clipping (bounding) as a stabilization mechanism for large delays. |
| `sweep_workers.py` | Phase-1 sweep over the number of Hogwild! async workers at fixed hyperparameters; primary output is a generalization-gap vs. worker-count plot that tests whether async parallelism itself acts as an implicit regularizer. |

---

### outputs/

Experimental results and generated figures. Each subdirectory corresponds to one experiment family.

| Directory | Description |
|---|---|
| `det_grid/` | Per-epoch CSV logs from the Step-3 deterministic delay × momentum grid sweep. |
| `det_theory/` | Per-epoch CSV logs from the Step-4 theory experiments (Exps A–H). |
| `geo_grid/` | Per-epoch CSV logs from the geometric-delay grid sweep (equivalent of `det_grid/`). |
| `geo_large/` | Per-epoch CSV logs from large-scale geometric-delay sweeps. |
| `geo_theory/` | Per-epoch CSV logs from the geometric-delay theory experiments. |
| `step5/` | Results from the main Hogwild!-style async-SGD experiments (Step 5). |
| `step5_ext/` | Extended Step-5 results including LR-rescue runs. |
| `grad_bound/` | Results from gradient-clipping stabilization experiments. |
| `async_experiments/` | Additional Hogwild! async-SGD experiment logs. |
| `delayed_experiments/` | Supplementary delayed-SGD experiment logs. |
| `seq_experiments/` | Sequential (no-delay, no-async) baseline experiment logs. |
| `figures/` | Publication-quality PDF and PNG figures generated by `make_figures.py`. |
| `plots/` | Exploratory analysis plots generated by the `analyze_*.py` scripts. |

---

## Root-Level Files

| File | Description |
|---|---|
| `environment.yml` | Conda environment specification (`OptML`, Python 3.11). Includes PyTorch, torchvision, scikit-learn, pandas, matplotlib, and Jupyter. |

---

## License

This project was developed for the Optimization for Machine Learning (CS-439) course at École Polytechnique Fédérale de Lausanne (EPFL). EPFL and the course instructors conserve all rights related to this project.

---

## Authors

- Aleksandar Mihaylov <aleksandar.mihaylov@epfl.ch>
- Alexandre Potocnik <alexandre.potocnik@epfl.ch>
- Ilia Badanin <ilia.badanin@epfl.ch>
