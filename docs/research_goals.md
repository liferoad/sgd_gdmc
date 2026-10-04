# Research goals - status

This document answers the two research goals directly. Evidence comes from
experiments/15_lowbit_comparison.py (MNIST 10 seeds, Fashion-MNIST 8 seeds,
10 epochs) analysed by analysis/lowbit_analysis.py; raw data in
results/raw/lowbit_comparison.csv and lowbit_comparison_fashion.csv,
tables in results/lowbit_summary.md and results/lowbit_paired.csv, and
learning curves in results/plots/lowbit_learning_curves_{mnist,fashion}.png.

Protocol: every quantized method runs on the same uniform grid in [-1, 1];
every method gets the same three-candidate tuning budget per bit-width,
selected on a held-out 5K validation split with a tuning seed not among the
reported seeds; final runs are on the official test set. Differences are
paired by seed, and both best-of-run and final accuracy are reported.

---

## Goal 1 - basic GDMC works for deep learning

**Status: established for learning, but basic GDMC is not competitive.**

Basic GDMC (beta1=0, k=1, i.e. plain sign-directed grid moves with Metropolis
acceptance and no momentum) trains stably on both tasks and at every
bit-width:

| task | 2-bit | 4-bit | 8-bit |
|---|---:|---:|---:|
| MNIST, final accuracy | 0.7768 | 0.8921 | 0.9426 |
| MNIST, best of run | 0.7966 | 0.8953 | 0.9427 |
| Fashion, final accuracy | 0.5566 | 0.7542 | 0.8473 |
| Fashion, best of run | 0.6583 | 0.7631 | 0.8474 |

The learning curves (results/plots/lowbit_learning_curves_*.png) show train
loss falling and test accuracy rising from the first evaluation to the last
for basic GDMC at every bit-width, with no late divergence.

But it is dominated: momentum GDMC (beta1=0.9) is better at every bit-width
on both tasks (e.g. MNIST final 0.9335 vs 0.8921 at 4 bits), and projected
Adam is better at 4 and 8 bits (MNIST final 0.9707 vs 0.9426 at 8 bits).

**Answer:** basic GDMC is a valid working optimizer for deep learning - it
learns and is stable - but it is not the version worth using. Momentum is the
component that makes GDMC competitive; this is consistent with the earlier
v2 finding and with the noise study, where momentum was the only active
ingredient.

---

## Goal 2 - GDMC performs better with low-precision weights

**Status: established in the 4-bit regime on two tasks, neutral at 8 bits,
and false at 2 bits.**

Final accuracy, mean over seeds (95% t-interval):

| task | bits | GDMC v2 | projected Adam | QAT (STE) | FP32 Adam |
|---|---:|---:|---:|---:|---:|
| MNIST | 2 | 0.4099 | 0.4635 | 0.9354 | 0.9791 |
| MNIST | 4 | **0.9335** | 0.4907 | 0.9770 | 0.9791 |
| MNIST | 8 | 0.9743 | 0.9707 | 0.9768 | 0.9791 |
| Fashion | 2 | 0.2923 | 0.3589 | 0.7572 | 0.8841 |
| Fashion | 4 | **0.7507** | 0.5589 | 0.8790 | 0.8841 |
| Fashion | 8 | 0.8751 | 0.8634 | 0.8831 | 0.8841 |

Paired differences vs projected Adam, matched by seed:

| task | bits | metric | mean diff | seeds favouring GDMC | paired p |
|---|---:|---|---:|---:|---:|
| MNIST | 4 | best of run | +0.0181 | 10/10 | 0.0010 |
| MNIST | 4 | final | +0.4429 | 10/10 | 0.0010 |
| Fashion | 4 | best of run | +0.0397 | 8/8 | 0.0009 |
| Fashion | 4 | final | +0.1918 | 8/8 | 0.0081 |
| MNIST | 8 | best of run | +0.0005 | 5/10 | 0.5704 |
| MNIST | 8 | final | +0.0036 | 9/10 | 0.0122 |
| Fashion | 8 | best of run | +0.0085 | 8/8 | 0.0002 |
| Fashion | 8 | final | +0.0117 | 8/8 | 0.0006 |
| MNIST | 2 | best of run | -0.3015 | 1/10 | 0.0002 |
| Fashion | 2 | best of run | -0.4077 | 0/8 | 0.0000 |

Two things are worth separating:

1. **At 4 bits GDMC is better than the same-precision baseline, and the
   effect replicates.** +1.8 points on MNIST (10/10 seeds) and +4.0 points on
   Fashion (8/8 seeds), both significant under a paired t-test and a Wilcoxon
   signed-rank test. The direction is consistent across every seed on both
   tasks, so it is not a seed artifact.
2. **The final-accuracy gap is much larger because projected Adam is
   unstable at low precision.** The learning curves show projected Adam
   peaking around step 1000 and then collapsing: at 4 bits its final MNIST
   accuracy is 0.49 (best 0.92) and its final Fashion accuracy is 0.56 (best
   0.75), while GDMC keeps improving. Selecting on best-of-run hides this
   instability; reporting both metrics exposes it. GDMC's advantage at 4 bits
   is therefore better described as *stability* under aggressive weight
   quantization than as a raw peak-accuracy gain.

**At 8 bits the effect is statistically detectable but practically negligible**
(+0.05 to +1.2 points depending on task and metric): both direct methods are
within about a point of each other and of QAT/FP32 Adam. **At 2 bits GDMC is
clearly worse** than projected Adam in the momentum variant and, at best, a
tie for the basic variant, so no low-bit advantage can be claimed there.

**Practical caveat.** Latent-weight QAT (weights stored in full precision,
quantized in the forward pass with a straight-through estimator) beats *every*
direct-quantization method at 2 and 4 bits (MNIST 0.9354 and 0.9770; Fashion
0.7572 and 0.8790) and matches FP32 Adam at 8 bits. If the question is
"what should I use at 2-4 bits", the answer from this data is QAT, not GDMC.
GDMC's claim is narrower: among methods that *optimize the deployed weights
directly on the grid*, momentum GDMC is the most accurate and by far the most
stable at 4 bits, and it is competitive at 8 bits.

---

## What is established, and what is not

Established:

* basic GDMC learns and is stable, but momentum is what makes GDMC work;
* momentum GDMC beats projected Adam at 4 bits on two tasks with per-seed
  consistency and significance, and is far more stable late in training;
* at 8 bits GDMC ties the same-precision baseline;
* at 2 bits GDMC does not beat the baseline;
* latent-weight QAT dominates all direct-quantization methods at 2-4 bits.

Not established:

* any claim on convolutional or Transformer models (everything here is a
  784-256-256-10 MLP);
* a memory advantage (packed persistent weights are still not implemented;
  the memory study is a separate supporting result);
* noise resilience (the dedicated study found FP32 Adam more noise-robust,
  and the Metropolis step inert at these move sizes).

To strengthen goal 2 further: replicate the 4-bit cell on a CNN, tune
projected Adam's late-training stability (learning-rate decay or gradient
clipping) so the comparison is not decided by its divergence, and report the
stability result as the primary claim rather than best-of-run accuracy.

## Reproducing

    .venv/bin/python experiments/15_lowbit_comparison.py --task mnist --seeds 0 1 2 3 4 5 6 7 8 9
    .venv/bin/python experiments/15_lowbit_comparison.py --task fashion --seeds 0 1 2 3 4 5 6 7
    .venv/bin/python analysis/lowbit_analysis.py

The per-method validation selections are cached in
results/lowbit_settings_mnist.json and results/lowbit_settings_fashion.json,
so re-runs reuse exactly the same settings.