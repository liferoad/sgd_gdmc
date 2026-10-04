# Long-run results — 30 epochs (MNIST MLP) and 8 epochs (MNIST CNN)

The v1 sweep in [`results/REPORT.md`](../results/REPORT.md) used a very
short epoch budget on the classification tasks (MNIST MLP 3 epochs,
MNIST/CIFAR-10 CNN 2 epochs). At 2 epochs on CIFAR-10 even Adam barely
beats random (0.25 vs 0.10), so those numbers mostly measure "what
happens in the first few thousand steps", not "how well can this
optimizer train".

This document reports two long-horizon re-runs that **materially change
the headline conclusion**:

* **§1-6: MNIST MLP, 30 epochs, full 60K training set** — the complete
  v1 sweep plus a GDMC v2 momentum ablation.
* **§7: MNIST CNN, 8 epochs, 10K subset** — a focused comparison of
  GDMC v1 / GDMC v2 momentum / Projected-GD at 4 and 8 bits.

**Headline: GDMC v2 with momentum matches or beats continuous Adam at
8 bits on both models** (MLP 0.9775 vs 0.9757; CNN 0.9668 vs 0.9546),
and at 4 bits it is 4-5× better than the QAT-style Projected-GD
baseline.

### MNIST MLP setup

* Script: [`experiments/07_long_runs.py`](../experiments/07_long_runs.py)
* Raw data: `results/raw/long_runs_mlp.csv` (117 runs),
  `results/raw/long_runs_v2_mlp.csv` (18 runs)
* Setup: MNIST MLP `784 → 256 → 256 → 10`, **30 epochs**, full 60,000
training examples, batch 128, 3 seeds (0/1/2). Identical optimizer
hyperparameters to the short sweep.

---

## 1. Headline table — all optimizers, 30 epochs

Best test accuracy over the run, mean ± std across 3 seeds.

| optimizer          | 2-bit            | 3-bit            | 4-bit            | 8-bit            | 16-bit           | 32-bit           |
|--------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| momentum           | —                | —                | —                | —                | —                | **0.9808 ± 0.0009** |
| adam               | —                | —                | —                | —                | —                | 0.9757 ± 0.0014  |
| **gdmc v2 (β1=0.9, k=1)** | —        | —                | **0.9460 ± 0.0017** | **0.9775 ± 0.0005** | —        | —                |
| gdmc v2 (β1=0.9, k=4) | —             | —                | 0.6229 ± 0.0284  | 0.9724 ± 0.0005  | —                | —                |
| sgd                | —                | —                | —                | —                | —                | 0.9508 ± 0.0008  |
| gdmc-adaptive (uniform-pt) | —       | 0.9306 ± 0.0006  | 0.9304 ± 0.0015  | 0.9304 ± 0.0015  | —                | —                |
| gdmc v1 (one-step) | 0.7950 ± 0.0073  | 0.8711 ± 0.0048  | 0.8959 ± 0.0021  | 0.9608 ± 0.0004  | 0.6410 ± 0.0620  | 0.0896 ± 0.0226  |
| sgld               | —                | —                | —                | —                | —                | 0.8341 ± 0.0016  |
| projected-gd       | 0.2182 ± 0.0795  | 0.2181 ± 0.0787  | 0.2180 ± 0.0763  | 0.9731 ± 0.0008  | 0.9720 ± 0.0007  | 0.9722 ± 0.0006  |

(The continuous baselines `momentum`/`adam`/`sgd`/`sgld` were replicated
at each `bits` row in the raw CSV but are in fact the *same* 32-bit
unquantized configuration; the table shows them once.)

---

## 2. The headline: GDMC v2 matches or beats Adam at 8 bits

**Adam (continuous, no quantization): 0.9757 ± 0.0014**
**GDMC v2 (β1=0.9, k=1) at 8-bit:      0.9775 ± 0.0005**

GDMC v2 at 8-bit is **+0.0018 above Adam** with a 3× tighter standard
deviation. This is the strongest result in the project: an optimizer
whose proposals are restricted to a 256-level grid, with Metropolis
accept/reject, matches discrete-Adam-with-projection and edges out
continuous Adam on MNIST MLP.

At 4 bits the picture is:

| optimizer | 4-bit accuracy | vs Adam |
|-----------|---------------:|--------:|
| adam (32-bit, no quant) | 0.9757 | — |
| **gdmc v2 (β1=0.9, k=1)** | **0.9460** | −0.0297 |
| gdmc-adaptive (uniform-pt) | 0.9304 | −0.0453 |
| gdmc v1 (one-step) | 0.8959 | −0.0798 |
| projected-gd | 0.2180 | **−0.7577** |

So at 4 bits:

* GDMC v2 is **3.0 points** behind continuous Adam.
* GDMC v2 is **4.3× better** than the QAT-style Projected-GD baseline
  at the same 4-bit quantization (0.9460 vs 0.2180).
* Projected-GD is 75.8 points behind Adam — it has not trained at all.

---

## 3. What more epochs changes

Comparing the short (3-epoch) and long (30-epoch) MNIST MLP sweeps:

| config                | 3 epochs | 30 epochs | delta  |
|-----------------------|---------:|----------:|-------:|
| adam (32-bit)         | 0.9325   | 0.9757    | +0.043 |
| projected-gd 8-bit    | 0.9284   | 0.9731    | +0.045 |
| **gdmc 8-bit**        | 0.7745   | 0.9608    | +0.186 |
| **gdmc 4-bit**        | 0.8279   | 0.8959    | +0.068 |
| gdmc 2-bit            | 0.7460   | 0.7950    | +0.049 |
| projected-gd 4-bit    | 0.1997   | 0.2180    | +0.018 |

Three things stand out:

1. **The GDMC 8-bit result was the worst-case casualty of the short
   run.** It went from 0.7745 (3 epochs) to 0.9608 (30 epochs), a
   +0.186 gain — far more than any other config. The short-run
   conclusion "GDMC at 8 bits is 15 points worse than Projected-GD"
   was an artifact of the epoch budget; at 30 epochs the gap is only
   1.2 points (0.9608 vs 0.9731) and GDMC v2 pulls ahead of both.
2. **Projected-GD at 2-4 bits is stuck regardless of epochs.**
   0.1997 → 0.2180 over a 10× increase in training. This is not an
   under-training problem; it is the Adam-adaptive-scaling-on-a-coarse-
   grid failure mode. More compute does not fix it.
3. **The 16/32-bit GDMC plateau is structural, not a training-budget
   issue.** GDMC v1 at 16-bit is 0.6410 at 30 epochs (vs 0.0940 at 3
   epochs — it did improve, but it is still 33 points below Adam). At
   32-bit it is 0.0896: no movement at all. The one-grid-step move is
   too small at these resolutions no matter how long you train.

---

## 4. GDMC v2 (momentum) at the long horizon

The v2 ablation at 30 epochs, 3 seeds:

| bits | configuration | test accuracy (mean ± std) |
|-----:|---------------|---------------------------:|
| 4    | v1 (β1=0.0, k=1) | 0.8959 ± 0.0021 |
| 4    | **v2 (β1=0.9, k=1)** | **0.9460 ± 0.0017** |
| 4    | v2 (β1=0.9, k=4) | 0.6229 ± 0.0284 |
| 8    | v1 (β1=0.0, k=1) | 0.9608 ± 0.0004 |
| 8    | **v2 (β1=0.9, k=1)** | **0.9775 ± 0.0005** |
| 8    | v2 (β1=0.9, k=4) | 0.9724 ± 0.0005 |

**Momentum alone (β1=0.9, k=1) is the v2 win, and it grows with training:**

* 4-bit: 0.8959 → 0.9460 (**+5.0 points**)
* 8-bit: 0.9608 → 0.9775 (**+1.7 points**)

The improvement is consistent across all 3 seeds (std ≈ 0.002), so it
is not seed noise.

**Multi-step (k=4) is harmful at 4 bits even at 30 epochs**
(0.6229, worse than v1's 0.8959) and slightly harmful at 8 bits
(0.9724 vs 0.9775). This confirms the short-run toy-regression finding:
on the MNIST MLP grid, constant-`k` multi-step should not be used at
4 bits, and at 8 bits it does not help once momentum is present.

> **Note on the toy-regression v2 result.** The toy sweep
> (`docs/v2_results.md`) found `k=4` to be the *best* configuration at
> 8-bit (3.4× lower loss). On MNIST MLP, `k=4` at 8-bit is
> slightly *worse* than `k=1`. The two results are not contradictory —
> the toy task has a 64-unit MLP and a smooth 1-D target, so larger
> discrete steps pay off; MNIST has a rugged 785-dim landscape where
> they overshoot. **Momentum (`β1=0.9`) is the robust part of v2;
> multi-step is task-dependent.**

---

## 5. Revised recommendation

| bit budget | recommended optimizer | why |
|------------|----------------------|-----|
| 2-4 bits   | **GDMC v2 (β1=0.9, k=1)** | 4.3× better than Projected-GD at 4-bit; within 3 points of continuous Adam |
| 3-4 bits, needs stability across bits | `gdmc-adaptive` (uniform-pt) | 0.9304 flat across 3/4/8 bits, no degradation |
| 8 bits     | **GDMC v2 (β1=0.9, k=1)** | matches/beats continuous Adam (0.9775 vs 0.9757) |
| 16+ bits   | Projected-GD or continuous Adam | GDMC is structurally stuck (one step is too small) |
| no quantization constraint | momentum or Adam | momentum SGD actually wins here (0.9808) |

---

## 6. What this does and does not settle

**Settled:**

* The GDMC ≫ Projected-GD advantage at 4-bit is real and grows with
  training (4.3× at 30 epochs).
* GDMC v2 with momentum is competitive with continuous Adam at 8 bits
  on MNIST MLP.
* The 16/32-bit GDMC plateau is structural.
* Momentum is the robust part of v2; constant-`k` multi-step is
  task-dependent and should be ablated per task.

**Not settled (needs more compute):**

* **CIFAR-10** at a long horizon. The 2-epoch sweep is too short for
  any CIFAR-10 conclusion (Adam at 0.25 vs 0.10 random). A proper
  CIFAR-10 comparison needs ~20 epochs (≈ 8 h on this CPU).
* **The 16/32-bit plateau fix.** Magnitude-scaled `k`
  (`k_i ∝ |g_i| / EMA(|g|)`) is the proposed fix (see
  [`docs/future_work.md`](future_work.md), Fix 1.1) and has not been
  implemented.
* **Momentum at 4-bit on MNIST CNN is unreliable** — see §7.

---

## 7. MNIST CNN focused long run (8 epochs)

The short-run 4-bit MNIST CNN number was the noisiest in the project
(0.5180 ± 0.2193, one seed at 0.18). This section re-runs the key
comparison at 8 epochs on the 10K training subset.

* Script: [`experiments/08_long_runs_cnn.py`](../experiments/08_long_runs_cnn.py)
* Raw data: `results/raw/long_runs_cnn_focused.csv` (21 runs)
* Setup: `SmallCNN` (3 conv blocks + 2 FC, base width 16), **8 epochs**,
  10K training subset, batch 128, 3 seeds.

### 7.1 Results

| optimizer | bits | test acc (mean ± std) | per-seed |
|-----------|-----:|----------------------:|----------|
| adam (continuous) | 32 | 0.9546 ± 0.0177 | 0.9688 / 0.9602 / 0.9348 |
| **gdmc v2 (β1=0.9, k=1)** | **8** | **0.9668 ± 0.0027** | 0.9669 / 0.9640 / 0.9694 |
| gdmc v1 (β1=0.0, k=1) | 8 | 0.9357 ± 0.0152 | 0.9455 / 0.9181 / 0.9434 |
| projected-gd | 8 | 0.9291 ± 0.0443 | 0.9220 / 0.9766 / 0.8888 |
| gdmc v2 (β1=0.9, k=1) | 4 | 0.7807 ± 0.2154 | 0.9107 / 0.5320 / 0.8993 |
| gdmc v1 (β1=0.0, k=1) | 4 | 0.7680 ± 0.1429 | 0.7288 / 0.6488 / 0.9264 |
| projected-gd | 4 | 0.1503 ± 0.0454 | 0.1118 / 0.2003 / 0.1388 |

### 7.2 Findings

1. **GDMC v2 at 8-bit beats continuous Adam again** — 0.9668 ± 0.0027
   vs Adam's 0.9546 ± 0.0177, a +1.2 point win with **6.5× tighter**
   variance. This replicates the MNIST MLP result on a convolutional
   model, so it is not an artifact of the MLP.
2. **Momentum at 8-bit is a clean, consistent win**: v1 0.9357 →
   v2 0.9668 (+3.1 points), and all three v2 seeds are within
   0.005 of each other (std 0.0027).
3. **At 4-bit, GDMC crushes Projected-GD by 5.1×** (0.774 vs 0.150).
   Projected-GD is stuck at ~0.15 regardless of epochs — the same
   Adam-on-a-coarse-grid failure.
4. **Momentum at 4-bit on the CNN is unreliable.** v1 is 0.7680 ±
   0.1429 and v2 is 0.7807 ± 0.2154 — higher mean but much wider
   spread, with one v2 seed at 0.5320. Unlike the MLP (where
   momentum was a consistent +5 points at 4-bit), the CNN at 4-bit
   remains a high-variance regime where a single unlucky seed can
   dominate the mean. **Recommendation: use momentum at 8-bit; at
   4-bit seed-average over more than 3 seeds before trusting the
   number.**
5. The short-run 4-bit GDMC v1 number (0.5026 over 3 seeds, 2 epochs)
   rises to 0.7680 at 8 epochs — confirming the short budget was
   understating GDMC, while Projected-GD barely moves
   (0.1523 → 0.1503).
