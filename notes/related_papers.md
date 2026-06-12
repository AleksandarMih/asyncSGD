# Related Papers — Notes for Mini-Project

---

## Mitliagkas et al. (2014) — "Asynchrony Begets Momentum"

**Reference:** Mitliagkas, I., Zhang, C., Smola, A., & Jordan, M. (2014). Asynchrony begets momentum, with an application to deep learning. *NIPS*.

### Core claim
With *k* async workers under a geometric staleness model (M/M/1 queue), the effective update
is equivalent to SGD with implicit momentum:

  µ_S = (k−1)/k

So `k=2` workers → `µ_S ≈ 0.5`, `k=10` → `µ_S ≈ 0.9`.

### Model assumptions (critical for interpreting our results)
- Staleness is **stochastic** (geometrically distributed, M/M/1 queue) — not deterministic.
- Workers truly run in parallel and unsynchronized (HogWild! setting).
- Section VI introduces the negative-momentum compensation idea:
  `β_explicit = β_target − (τ−1)/τ` → pushed to negative values at large τ.

### Why our Exp A failed
Our `train_step4.py` uses a **FIFO deterministic buffer** with fixed lag τ — a single-process
simulation, not a real async system. The equivalence is derived for *geometric* staleness.
Deterministic delay has different spectral properties; the implicit momentum effect may not appear.

### What to cite for the report
- The µ_S = (k−1)/k formula (paper Section III/IV).
- Section VI as motivation for Exp F (negative explicit momentum).
- That the equivalence requires stochastic staleness — our Exp A tests this boundary.

---

## DANA — Hakimi et al. (2020)

**Reference:** Hakimi, S., Elesedy, B., & Kratsios, A. (2020). *DANA: Distortion Correction for
Adaptive Networks*. arXiv:1907.11612.

### Problem addressed
Momentum (Nesterov/MSGD) combined with asynchrony drastically increases gradient staleness.
The gap metric (not lag) quantifies this effect.

### The gap metric
  G(Δ_{t+τ}) = RMSE(θ_{t+τ} − θ_t) = ‖θ_{t+τ} − θ_t‖₂ / √k

- More informative than lag τ alone because it captures the actual parameter-space drift.
- Gap correlates **linearly with learning rate** (η); drops sharply at LR decay milestones.
- NAG-ASGD gap >> ASGD gap at the same lag: momentum amplifies staleness in parameter space.

### DANA algorithm
**DANA-Zero:** Master maintains a separate momentum buffer per worker `v^i`. Instead of
sending current params `θ`, sends a look-ahead estimate:
  θ̂ = θ − ηγ Σ_j v^j
Key theorem: `E[Δ^DANA] = E[Δ^ASGD]` — gap is reduced back to vanilla ASGD level despite
using momentum.

**DANA-Slim:** Bengio-NAG reformulation. Worker sends `γv^i + g^i` instead of just `g^i`;
no per-worker momentum storage at master. Algorithmically equivalent to DANA-Zero, zero
overhead. This is the practical version.

### Empirical results (Table 2) — ResNet-20 on CIFAR-10
Same model and dataset as ours. **Baseline (1 worker): 91.63%.**

| Method        | 4W     | 8W     | 12W    | 16W    | 20W    | 32W    |
|---------------|--------|--------|--------|--------|--------|--------|
| DANA-DC       | 91.50% | 91.27% | 90.97% | 90.71% | 90.07% | 82.99% |
| DC-ASGD       | 91.44% | 91.20% | 90.55% | —      | FAIL   | FAIL   |
| NAG-ASGD      | 91.41% | 91.18% | 90.77% | FAIL   | FAIL   | FAIL   |
| Multi-ASGD    | 91.40% | 91.17% | 90.63% | 90.34% | FAIL   | FAIL   |
| SSGD (sync)   | 91.63% | 91.50% | 91.22% | 90.71% | 89.93% | 87.28% |

**Hyperparameters (identical to ours):** lr=0.1, batch=128, weight_decay=1e-4,
LR decay at epochs 80 and 120 (160 total epochs).

### Future experiments / implementation ideas
1. **Measure the gap metric** in our HogWild! Step 5 runs. Log `‖θ_{t+1} − θ_t‖₂` each
   epoch per worker; compare gap across k=2,4,8. Does gap grow with k?
2. **Implement DANA-Slim** on top of our HogWild! shared-memory implementation. The change
   is in the worker: compute gradient on `θ̂ = θ − ηγv^i` instead of `θ`, then send `γv^i + g`.
3. **Compare our accuracy numbers** to DANA Table 2. At 50 epochs our ceiling is lower, but
   the relative degradation vs. k should be comparable.
4. DC-ASGD (delay compensation via second-order Taylor) as a simpler baseline to implement:
   compensated gradient `g_comp = g + λ · ∇²L · (θ_t − θ_{t−τ}) · g`.

---

## Liu et al. (NeurIPS 2018)

**Reference:** Liu, X., Pan, W., Shen, L., & Gu, B. (2018). Towards Understanding Acceleration
Tradeoff between Momentum and Asynchrony in Nonconvex Stochastic Optimization. *NeurIPS*.

### Core theoretical result
For Async-MSGD to converge on a nonconvex strict-saddle problem, the delay must satisfy:

  τ ≲ (1 − µ)² / η^{1−γ}    (for some γ ∈ (0, 0.5])

This is the **fundamental momentum-asynchrony tradeoff**: higher momentum → lower allowable delay.

### Implication for our setting (η=0.1, µ=0.9)
  τ_max ≈ (1−0.9)² / 0.1^{0.5} ≈ 0.01 / 0.316 ≈ 0.032

Even delay τ=1 exceeds the theoretical bound for µ=0.9. This is entirely consistent with
our Step 3 results where β=0.9 degrades badly even at τ=2.

For delay τ=10 to be stable, we need:
  (1−µ)² ≳ τ × η ≈ 10 × 0.1 = 1  →  µ ≲ 0

**Negative explicit momentum is theoretically predicted to be necessary at large delays.**
This directly motivates Exp F.

### Remark 7 — explicit refutation of Mitliagkas equivalence
> "Mitliagkas et al. (2016) conjecture that the delay in Async-SGD is equivalent to the
> momentum in MSGD. Our result, however, shows that this is not true in general."

This is the clearest available citation for why Exp A (the Mitliagkas equivalence test)
yielded a negative result. The equivalence is not general.

### Figure 2 — optimal µ vs. number of workers (DNN experiment)
| Workers | Optimal µ |
|---------|-----------|
| 1       | 0.9       |
| 2       | 0.9       |
| 4       | 0.7       |
| 8       | 0.5       |

"µ=0.9 yields the worst performance for τ=8." — direct empirical confirmation.
The monotone decrease in optimal µ with k supports our hypothesis in Exp F.

### Figure 1 — optimal delay vs. µ (streaming PCA experiment)
| µ     | Optimal τ |
|-------|-----------|
| 0.70  | 120       |
| 0.80  | 80        |
| 0.85  | 60        |
| 0.90  | 30        |
| 0.95  | 10        |

Lower momentum tolerates much higher delay — the (1−µ)² bound is reflected empirically.

### Future experiments / things to test
1. **Validate the (1−µ)² bound empirically.** For each τ ∈ {2, 5, 10, 20}, find the µ
   at which accuracy begins to degrade. Does the threshold follow µ_crit ≈ 1 − √(τ·η)?
2. **Optimal µ sweep.** Match Figure 2: for each number of HogWild! workers k ∈ {2, 4, 8},
   do a fine µ sweep (0.0, 0.3, 0.5, 0.7, 0.9) and find the empirical optimum.
   Predicted: optimum shifts from ~0.9 at k=1 toward ~0.5 at k=8.
3. **Negative µ validation.** If (1−µ)² must exceed τ·η, and τ is large, µ<0 is needed.
   Exp F tests exactly this. After Exp F runs, check: does the accuracy peak at a µ
   consistent with µ_crit = 1 − √(τ·η)?

### What to cite for the report
- Section 4 Theorem for the (1−µ)² convergence bound.
- Remark 7 for the Mitliagkas equivalence refutation.
- Figure 2 to motivate Exp F and to validate any results we obtain there.
- Liu et al. as theoretical grounding for Step 3 empirical observation that β=0.9 fails
  at any nonzero delay.

---

## Deng et al. (IEEE TPAMI 2025) — "Toward Understanding the Generalizability of Delayed SGD"

**Reference:** Deng, X., Shen, L., Li, S., Sun, T., Li, D., & Tao, D. (2025). Toward Understanding
the Generalizability of Delayed Stochastic Gradient Descent. *IEEE TPAMI*, Vol. 47, No. 9.

### Core claim
Asynchronous delay **reduces** generalization error — counter-intuitive but theoretically proven.
The mechanism is increased algorithmic stability: a model trained with delay is less sensitive
to any individual training sample.

### Bounds
- Convex quadratic: generalization error ≤ O((T−τ)/(nτ)) — **improves with τ**
- Strongly convex: O(1/n) — independent of T and τ entirely
- Requires learning rate η ≤ 1/(20μ(τ+1)) — safe LR shrinks with delay
- Extends to bounded random delays (Corollary 1) — same O((T−τ̄)/(nτ̄)) bound

### What this explains in our results

**Train-test gap stays constant (our plot_Q1_train_test_gap).**
We observed the gap doesn't grow with τ and noted "no implicit regularisation."
This paper provides the theoretical reason: delay increases stability → gap should not grow,
and may slightly shrink. Our result is consistent with their theory.
Cite: "consistent with Deng et al. (2025) who prove that delay improves algorithmic stability."

**Theoretical grounding for LR rescue.**
Their safe LR requirement η ≤ 1/(20μ(τ+1)) directly motivates scaling η → η/τ at large delays.
The penalty for violating this bound (instability/poor generalisation) is exactly what we observe
at τ=64 without LR correction. Cite as motivation for the LR rescue sweep.

**Random delay / geometric setting.**
Section VI covers bounded random delays — geometric delay satisfies their Assumption 4 (bounded τ_t ≤ τ̄).
The same stability improvement holds, supporting our claim that geometric delay is "more stable"
than deterministic at the same E[τ].

### Empirical validation in the paper
- ResNet-18 on CIFAR-100 (Fig. 1, Fig. 6): generalization error decreases monotonically with τ ∈ {4,8,16,32}
- FC+MNIST, quadratic LIBSVM datasets: same trend
- Used fixed LR (no step decay), 16 distributed workers, 5 seeds

### Caveats
- Theory is for quadratic convex loss; NTK argument used to claim DNN extension (speculative)
- Fixed LR setting — our LR milestone sensitivity at epoch 30 is outside their framework
- Their "generalization error" = train loss − test loss (not accuracy gap); direction consistent

### What to cite in the report
- Q1 Discussion (after train-test gap observation): cite for why the gap doesn't grow
- LR rescue subsection: cite as theoretical motivation for η → η/(τ+1) scaling
- Geometric delay discussion: cite Corollary 1 for random delay stability result

---

## Connections to Our Experiments

| Our Experiment | Paper connection |
|----------------|-----------------|
| Step 3 — τ×β grid (β=0, 0.5, 0.9) | Liu et al. Thm: β=0.9 unstable ∀ τ>0; empirically confirmed |
| Exp A — equivalence (τ=k, β=0) vs (τ=0, β=(k-1)/k) | Mitliagkas claim; Liu Remark 7: not true in general |
| Exp B — compensation β_comp = 0.9 − β_impl | Mitliagkas §VI; partial fix; doesn't go negative |
| Exp C — LR scaling 1/√τ | Standard async-SGD heuristic; not covered in these papers |
| Exp F — negative explicit momentum | Mitliagkas §VI + Liu Thm: directly predicted for large τ |
| Step 5 — HogWild! k-worker vs. sequential β=(k-1)/k | Mitliagkas M/M/1 equivalence in the real stochastic setting |
| (Future) gap metric | DANA Fig 2: gap reveals staleness better than lag |
| (Future) DANA-Slim | DANA Table 2: maintains >90% to 16 workers where NAG-ASGD fails |
| (Future) optimal µ vs k sweep | Liu Fig 2: monotone decrease; verifiable with HogWild! setup |
| Q1 train-test gap constant across τ | Deng et al. 2025: delay → stability → gap should not grow |
| Exp C — LR rescue η→η/τ | Deng et al. 2025: safe LR ≤ 1/(20μ(τ+1)) — direct motivation |
| Geo delay more stable than FIFO | Deng et al. 2025 Corollary 1: random delay has same stability bound |
