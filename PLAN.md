# PLAN.md — asyncSGD Project Wrap-Up

> **Living document.** Mark tasks ✅ as they complete.
> Last updated: 2026-06-08.

---

## Current State — Summary

All training experiments are complete. The report has been substantially
rewritten. One analysis job (`step3-analyze-v1`) is running to regenerate
`plot_C` with the new three-line design. Everything else is done.

---

## 1. Experiments — All Complete ✅

### Deterministic (FIFO) delay

| Exp | Description | Data | Status |
|-----|-------------|------|--------|
| Step 3 det. | τ × β grid: τ∈{0,2,5,10,20} × β∈{0,0.5,0.9} × 3 seeds | `outputs/step3/` | ✅ |
| Exp A det. | Mitliagkas equivalence: (τ=k,β=0) vs (τ=0,β=µ_S) | `outputs/step4/expA_*.csv` | ✅ |
| Exp F det. | Negative momentum: β∈[−0.9,0] × τ∈{5,10,20} | `outputs/step4/expF_*.csv` | ✅ |
| Exp G det. | Adaptive Liu-bound schedule × τ∈{5,10,20} × 4 schedules | `outputs/step4/expG_*.csv` | ✅ |
| Exp H det. | Sweet-spot δ sweep above Liu bound × τ∈{5,10,20} | `outputs/step4/expH_*.csv` | ✅ |

### Geometric (stochastic) delay

| Exp | Description | Data | Status |
|-----|-------------|------|--------|
| Step 3 geo. | M × β grid: M∈{1,3,6,11,21} × β∈{0,0.5,0.9} × 3 seeds | `outputs/step3_geo/` | ✅ |
| Exp A geo. | Mitliagkas equivalence under geometric delay: M∈{3,6} | `outputs/step3_geo/expA_*.csv` | ✅ |
| Exp G geo. | Adaptive schedule under geometric delay: M∈{3,6,11} × 4 schedules | `outputs/step3_geo/expG_*.csv` | ✅ |

---

## 2. Plots — Status

| Plot file | Generates from | In report | Status |
|-----------|---------------|-----------|--------|
| `plot_C_momentum_benefit.png` | `analyze_step3.py` | Fig. 1 (main) | ⏳ job `step3-analyze-v1` running |
| `plot_L2_adaptive_summary.png` | `analyze_step4.py` | Fig. 2 (main) | ✅ |
| `plot_geo_step3_comparison.png` | `analyze_geometric.py` | Fig. 3 (main) | ✅ |
| `plot_K_negative_momentum.png` | `analyze_step4.py` | Appendix | ✅ |
| `plot_L1_adaptive_curves.png` | `analyze_step4.py` | Appendix | ✅ |
| `plot_M_above_bound_sweep.png` | `analyze_step4.py` | Appendix | ✅ |
| `plot_geo_equivalence.png` | `analyze_geometric.py` | Appendix | ✅ |
| `plot_geo_adaptive.png` | `analyze_geometric.py` | Appendix | ✅ |

All plots live in `outputs/step4/`.

---

## 3. Report — Status

File: `report/report.tex`

| Section | Owner | Status |
|---------|-------|--------|
| Abstract — Q1 (convergence) | Ilia | ⬜ placeholder |
| Abstract — Q2 (momentum × delay) | Alex | ✅ written |
| Abstract — Q3 (regularization) | Aleksandar | ⬜ placeholder |
| Introduction Q1 contributions | Ilia | ⬜ placeholder |
| Introduction Q3 contributions | Aleksandar | ⬜ placeholder |
| Methods §2.1 Common Setup | Alex | ✅ |
| Methods §2.2 Delay Simulation | Alex | ✅ |
| Methods Q1 setup | Ilia | ⬜ placeholder |
| Methods Q3 setup | Aleksandar | ⬜ placeholder |
| Results §3.1 Deterministic delay | Alex | ✅ |
| Results §3.2 Geometric vs deterministic | Alex | ✅ |
| Results Q1 subsection | Ilia | ⬜ placeholder |
| Results Q3 subsection | Aleksandar | ⬜ placeholder |
| Discussion — momentum/theory | Alex | ✅ |
| Discussion — simulation limitations | Alex | ✅ |
| Discussion Q1 paragraph | Ilia | ⬜ placeholder |
| Discussion Q3 paragraph | Aleksandar | ⬜ placeholder |
| Summary | Alex + teammates | ⬜ partial |

---

## 4. Remaining Tasks

### Immediate (Alex)
- [ ] Verify `step3-analyze-v1` succeeded: check `outputs/step4/plot_C_momentum_benefit.png` looks right
- [ ] Copy final plots into `report/` directory (or update `\graphicspath` in report.tex so LaTeX can find them)
- [ ] Compile PDF: `cd report && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`

### Teammates
- [ ] **Ilia**: fill all `% [Q1 TEAMMATE Ilia: ...]` placeholders in `report/report.tex`
- [ ] **Aleksandar**: fill all `% [Q3 TEAMMATE Aleksandar: ...]` placeholders in `report/report.tex`

### Final review (everyone)
- [ ] Read compiled PDF end-to-end — check figure references, caption accuracy, page limit
- [ ] Verify all numbers cited in text match the actual plots
- [ ] Check bibliography (`report/literature.bib`) is complete

---

## 5. Rename and Cleanup Plan ✅ COMPLETE

The current naming is a historical accident (`step3`, `step4`, `step3_geo`) and does
not reflect the two-axis structure of the project (det. vs geo, grid vs theory).
`step3_geo` is especially confusing because it contains both the geo grid sweep
*and* the geo theory experiments (Exp A geo, Exp G geo) that logically belong
alongside `step4`.

### 5a. Target directory layout

```
asyncSGD/
├── src/                          (unchanged)
├── scripts/                      (renamed — see §5b)
├── report/                       (unchanged)
├── outputs/
│   ├── det_grid/                 (was step3/)       τ×β CSVs, plot_A/B/C
│   ├── det_theory/               (was step4/ CSVs)  Exp A–H CSVs
│   ├── geo_grid/                 (was step3_geo/ expA_geo_M* files)
│   ├── geo_theory/               (was step3_geo/ expG_geo_M* files)
│   └── plots/                    (was step4/ PNGs)  all generated figures
└── notes/
```

### 5b. Script renames

| Current name | New name | Reason |
|---|---|---|
| `train_step4.py` | `train.py` | single entry-point for all delay experiments |
| `analyze_step3.py` | `analyze_det_grid.py` | matches new output dir name |
| `analyze_step4.py` | `analyze_det_theory.py` | matches new output dir name |
| `analyze_geometric.py` | `analyze_geo.py` | shorter, consistent |
| `sweep_step3.sh` | `sweep_det_grid.sh` | descriptive |
| `sweep_step4.sh` | `sweep_det_theory.sh` | descriptive |
| `sweep_expF.sh` | `sweep_neg_momentum.sh` | descriptive |
| `sweep_expF2.sh` | `sweep_neg_momentum_ext.sh` | descriptive |
| `sweep_expG.sh` | `sweep_adaptive_det.sh` | descriptive |
| `sweep_expH.sh` | `sweep_above_bound.sh` | descriptive |
| `sweep_step3_geometric.sh` | `sweep_geo_grid.sh` | descriptive |
| `sweep_expA_geometric.sh` | `sweep_geo_equivalence.sh` | descriptive |
| `sweep_expG_geometric.sh` | `sweep_adaptive_geo.sh` | descriptive |

Notebooks (`data_exploration.ipynb`, `train_seq_baseline.ipynb`,
`train_seq_baseline_ref.ipynb`) — delete, not used in paper.

### 5c. Code changes required after renaming

**Output directory paths** — update `--out-dir` / `--in-dir` defaults in each script:

| Script | Argument | Old default | New default |
|---|---|---|---|
| `analyze_det_grid.py` | `--in-dir`, `--out-dir` | `outputs/step3` | `outputs/det_grid` → plots to `outputs/plots` |
| `analyze_det_theory.py` | `--step3-dir` | `outputs/step3` | `outputs/det_grid` |
| `analyze_det_theory.py` | `--step4-dir`, `--out-dir` | `outputs/step4` | `outputs/det_theory` → plots to `outputs/plots` |
| `analyze_geo.py` | `--det-dir` | `outputs/step3` | `outputs/det_grid` |
| `analyze_geo.py` | `--geo-dir` | `outputs/step3_geo` | `outputs/geo_grid` + `outputs/geo_theory` |
| `analyze_geo.py` | `--out-dir` | `outputs/step4` | `outputs/plots` |
| `train.py` | `--out-dir` | `outputs/step4` | `outputs/det_theory` or `outputs/geo_theory` |
| All sweep scripts | `--out-dir` in `COMMON=` | `outputs/step3` or `outputs/step4` | matching new dirs |

**Training script reference** — sweep scripts call `train_step4.py` by name; update to `train.py`.

Note: `df["experiment"] == "step3"` in `analyze_det_theory.py` is a CSV column value,
not a path — it does not change.

### 5d. Execution order for the rename

Do this in one go after the report is finalised (avoids breaking running jobs mid-edit):

```bash
cd /mnt/course-ee-559/rcp-caas-ee-559-g44/scratch-g44/asyncSGD

# 1. Create new output directories
mkdir -p outputs/det_grid outputs/det_theory outputs/geo_grid outputs/geo_theory outputs/plots

# 2. Move det grid data (CSVs only, not plots)
mv outputs/step3/tau*.csv outputs/det_grid/

# 3. Move det theory CSVs
mv outputs/step4/exp*.csv outputs/det_theory/

# 4. Split geo data by experiment type
mv outputs/step3_geo/expA_geo_*.csv outputs/geo_grid/    # M×β grid
mv outputs/step3_geo/expG_geo_*.csv outputs/geo_theory/  # adaptive schedule

# 5. Move all plots to outputs/plots/
mv outputs/step3/plot_*.png outputs/det_grid/   # plot_A, plot_B, plot_C stay with det_grid
mv outputs/step4/plot_*.png outputs/plots/

# 6. Rename scripts
mv scripts/train_step4.py        scripts/train.py
mv scripts/analyze_step3.py      scripts/analyze_det_grid.py
mv scripts/analyze_step4.py      scripts/analyze_det_theory.py
mv scripts/analyze_geometric.py  scripts/analyze_geo.py
mv scripts/sweep_step3.sh        scripts/sweep_det_grid.sh
mv scripts/sweep_step4.sh        scripts/sweep_det_theory.sh
mv scripts/sweep_expF.sh         scripts/sweep_neg_momentum.sh
mv scripts/sweep_expF2.sh        scripts/sweep_neg_momentum_ext.sh
mv scripts/sweep_expG.sh         scripts/sweep_adaptive_det.sh
mv scripts/sweep_expH.sh         scripts/sweep_above_bound.sh
mv scripts/sweep_step3_geometric.sh  scripts/sweep_geo_grid.sh
mv scripts/sweep_expA_geometric.sh   scripts/sweep_geo_equivalence.sh
mv scripts/sweep_expG_geometric.sh   scripts/sweep_adaptive_geo.sh

# 7. Delete unused notebooks
rm scripts/data_exploration.ipynb \
   scripts/train_seq_baseline.ipynb \
   scripts/train_seq_baseline_ref.ipynb

# 8. Apply path and script-name changes in code (see §5c table above)

# 9. Smoke test — regenerate plots from scratch
python3 scripts/analyze_det_grid.py
python3 scripts/analyze_det_theory.py
python3 scripts/analyze_geo.py
```

---

## 6. Active Scripts (current names, pre-rename)

```
src/
  model.py              ResNet-20
  data.py               CIFAR-10 loaders

scripts/
  train_step4.py        Core training: FIFO + geometric delay, all experiments
  analyze_step3.py      Plots for det. grid (plot_A, plot_B, plot_C)
  analyze_step4.py      Plots for det. theory (plot_D through plot_M)
  analyze_geometric.py  Plots for geo experiments (plot_geo_*)
  sweep_step3.sh        Det. grid sweep
  sweep_step4.sh        Det. theory Exp A–E sweep
  sweep_expF.sh         Exp F: negative momentum
  sweep_expF2.sh        Exp F extended (β=-0.7,-0.9 at τ=20)
  sweep_expG.sh         Exp G: adaptive schedule det.
  sweep_expH.sh         Exp H: sweet-spot δ sweep
  sweep_step3_geometric.sh   Geo grid sweep
  sweep_expA_geometric.sh    Geo equivalence (Exp A geo)
  sweep_expG_geometric.sh    Geo adaptive schedule (Exp G geo)
  data_exploration.ipynb        → DELETE
  train_seq_baseline.ipynb      → DELETE
  train_seq_baseline_ref.ipynb  → DELETE

report/
  report.tex            Main paper
  literature.bib        Bibliography

outputs/
  step3/           Det. grid CSVs + plot_A/B/C
  step4/           Det. theory CSVs + ALL generated plots
  step3_geo/       Geo grid + geo theory CSVs (mixed — split during rename)
```

---

## 7. Reproducibility Checklist

- [ ] All seeds fixed: {42, 123, 456}; `torch.backends.cudnn.deterministic = True`
- [ ] No absolute paths in any script (all relative to repo root)
- [ ] `environment.yml` tested from scratch
- [ ] PDF compiles without errors or missing references
- [ ] `.gitignore` excludes: `data/`, `outputs/*/` CSVs, `__pycache__/`
