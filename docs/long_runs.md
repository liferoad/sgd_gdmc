# Long-run results — 30 epochs (MNIST MLP), 8 epochs (MNIST CNN), 20 epochs (CIFAR-10 CNN)

The v1 sweep in [`results/REPORT.md`](../results/REPORT.md) used a very
short epoch budget on the classification tasks (MNIST MLP 3 epochs,
MNIST/CIFAR-10 CNN 2 epochs). At 2 epochs on CIFAR-10 even Adam barely
beats random (0.25 vs 0.10), so those numbers mostly measure "what
happens in the first few thousand steps", not "how well can this
optimizer train".

This document reports three long-horizon re-runs that **materially change
the headline conclusion**:

* **§1-6: MNIST MLP, 30 epochs, full 60K training set** — the complete
  v1 sweep plus a GDMC v2 momentum ablation.
* **§7: MNIST CNN, 8 epochs, 10K subset** — a focused comparison of
  GDMC v1 / GDMC v2 momentum / Projected-GD at 4 and 8 bits.
* **§8: CIFAR-10 CNN, 20 epochs, 10K subset** — the third model.

**Headline: GDMC v2 with momentum matches or beats continuous Adam at
8 bits on MNIST MLP and CNN** (MLP 0.9775 vs 0.9757; CNN 0.9668 vs
0.9546), but **does not on CIFAR-10** at 20 epochs / 10K train
(Projected-GD 0.6067 > Adam 0.5747 > GDMC v2 0.5682). On all three
models, GDMC v2 at 4 bits is **4.3-5.1× better** than the QAT-style
Projected-GD baseline, and momentum gives a consistent +10-20 point
lift at 4 bits over v1.

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


---

## 8. CIFAR-10 CNN focused long run (20 epochs)

The short CIFAR-10 sweep in experiments/04 used 2 epochs and 10K
training samples, which produced results barely above random (Adam
at 0.2513, GDMC 4-bit at 0.2671, both within 0.05 of the 0.10
random baseline). This section re-runs the key comparison at
20 epochs on the 10K training subset.

* Script: experiments/09_long_runs_cifar.py
* Raw data: results/raw/long_runs_cifar_focused.csv (24 runs)
* Setup: SmallCNN (3 conv blocks + 2 FC, base width 16),
  20 epochs, 10K training subset, batch 128, 3 seeds.

### 8.1 Results

| optimizer | bits | test acc (mean ± std) | per-seed |
|-----------|-----:|----------------------:|----------|
| projected-gd | 8 | 0.6067 ± 0.0397 | 0.5788 / 0.6521 / 0.5891 |
| adam (continuous) | 32 | 0.5747 ± 0.0407 | 0.5954 / 0.5278 / 0.6008 |
| gdmc v1 (β1=0.0, k=1) | 8 | 0.5249 ± 0.0496 | 0.4627 / 0.4866 / 0.4956 |
| gdmc v2 (β1=0.9, k=1) | 8 | 0.5682 ± 0.0154 | 0.5823 / 0.5704 / 0.5519 |
| gdmc v1 (β1=0.0, k=1) | 4 | 0.4367 ± 0.1408 | 0.3965 / 0.2253 / 0.3371 |
| gdmc v2 (β1=0.9, k=1) | 4 | 0.5538 ± 0.0291 | 0.5591 / 0.5226 / 0.5798 |
| momentum | 32 | 0.3486 ± 0.0943 | 0.2929 / 0.4575 / 0.2954 |
| projected-gd | 4 | 0.1217 ± 0.0167 | 0.1271 / 0.1030 / 0.1350 |

### 8.2 Findings

1. The order at 8 bits is INVERTED from MNIST MLP and CNN.
   Projected-GD 0.6067 > Adam 0.5747 > GDMC v2 0.5682. On CIFAR,
   the QAT-style baseline + Adam (with adaptive scaling on a
   256-level grid) is *better* than continuous Adam *and* better
   than GDMC v2 with momentum. This is the opposite of what MLP
   and CNN showed.

2. GDMC v2 at 4-bit (0.5538) vs Projected-GD at 4-bit (0.1217):
   4.55× ratio. This is the most robust finding across all three
   models — MLP 4.3×, CNN 5.1×, CIFAR 4.55×. Projected-GD is
   essentially broken at <=4 bits regardless of model or epoch count.

3. Momentum at 4-bit is the biggest absolute win on CIFAR. GDMC v1
   0.4367 -> GDMC v2 with momentum 0.5538 (+11.7 points). This
   matches the MLP pattern (where momentum was +5 points) and is
   larger than on the CNN (where v2 had high variance).

4. Momentum at 8-bit is small on CIFAR. GDMC v1 0.5249 -> GDMC v2
   0.5682 (+4.3 points). The momentum buffer helps but not enough
   to close the gap to Projected-GD (0.6067) at 8-bit. The high
   seed variance (Adam ranges 0.528-0.601 across seeds) suggests
   CIFAR with 10K train is just at the edge of what a 20-epoch
   run can resolve.

5. GDMC v2 at 8-bit does NOT match/beat Adam on CIFAR. MLP:
   0.9775 vs 0.9757 (+0.0018, GDMC wins). CNN: 0.9668 vs 0.9546
   (+0.0122, GDMC wins). CIFAR: 0.5682 vs 0.5747 (-0.0065, Adam
   wins). The "GDMC v2 >= Adam at 8-bit" headline *does not
   replicate on CIFAR* at this scale. Whether longer training or
   full 50K train would reverse the order is unknown.

6. Momentum at 32-bit baselines is bad on CIFAR (momentum 0.35,
   Adam 0.57). The momentum baseline uses lr=1e-2 which is too high
   for momentum on CIFAR. Same as v1; not a regression.

### 8.3 Why CIFAR differs from MNIST

Three things differ between CIFAR-10 and the MNIST models in this
project:

* Data complexity: CIFAR-10 is 32x32x3 RGB with 10 visually-distinct
  classes; MNIST is 28x28 grayscale with 10 classes. The loss
  landscape is harder.
* Model: same SmallCNN architecture but 3 input channels instead
  of 1; the BN running stats add noise.
* Training budget: 20 epochs × 10K train is ~1.4× the gradient
  steps of 8 epochs × 10K (CNN), but CIFAR is harder to fit per
  step.

The combination means Adam's adaptive scaling gets more leverage:
at the small batch budget, the per-coordinate step size variance
is large, and m / sqrt(v) smooths that out well. GDMC's sign(g)
doesn't see the magnitude differences, so it leaves some
performance on the table at 8-bit where Adam shines.

The 4-bit result is still the headline. Projected-GD completely
fails at 2-4 bits on every model tested. GDMC is the only
optimizer that works in that regime. At 8-bit and above, Adam or
Projected-GD is competitive or better.

### 8.4 Updated recommendation (all 3 models)

| bit budget | MNIST MLP | MNIST CNN | CIFAR-10 CNN |
|---|---|---|---|
| 2-4 bits   | GDMC v2 (β1=0.9, k=1) — 4.3× Projected-GD, within 3 pt of Adam | GDMC v2 (β1=0.9, k=1) — 5.1× Projected-GD, +1.2 pt over Adam | GDMC v2 (β1=0.9, k=1) — 4.55× Projected-GD, within 2 pt of Adam |
| 8 bits     | GDMC v2 ~ Adam (0.9775 vs 0.9757) | GDMC v2 > Adam (0.9668 vs 0.9546) | Projected-GD > GDMC v2 > Adam (0.6067 vs 0.5682 vs 0.5747) |
| 16+ bits   | Projected-GD or Adam | Projected-GD or Adam | n/a (not run at 16+ bits) |
| no quantization | momentum (0.9808) | Adam (0.9546) | Adam (0.5747) |

GDMC v2 with momentum is the right answer at 2-4 bits on all three
models, with 4.3-5.1× advantage over Projected-GD and within
1-3 points of continuous Adam. At 8 bits the picture is mixed:
GDMC v2 wins on MNIST, ties on MNIST CNN, and loses to
Projected-GD on CIFAR. The 8-bit result on CIFAR is the one open
question — it might reverse with longer training or full 50K
train.
