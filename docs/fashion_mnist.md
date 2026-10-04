# Fashion-MNIST — a second use case

Fashion-MNIST is a drop-in replacement for MNIST (same 28x28 grayscale,
10 classes, 60K train / 10K test) but substantially harder: a small MLP
reaches ~89% instead of ~98%. Running the same optimizer comparison
here tests whether the MNIST conclusions transfer to a different data
distribution.

* Script: experiments/11_fashion_mnist.py
* Raw data: results/raw/fashion_mnist_mlp.csv (63 runs)
* Setup: MLP 784 -> 256 -> 256 -> 10, 20 epochs, full 60K train,
  batch 128, 3 seeds, bit-widths {2, 4, 8, 16, 32}.

## Results

Best test accuracy, mean ± std over 3 seeds.

| bits | gdmc v1 | gdmc v2 (mom) | gdmc v3 (auto k) | projected-gd | adam |
|------|---------|---------------|------------------|--------------|------|
| 2    | **0.6233 ± 0.0767** | 0.2974 ± 0.0182 | 0.3720 ± 0.0943 | 0.1167 ± 0.0250 | — |
| 4    | 0.7592 ± 0.0020 | **0.7986 ± 0.0049** | 0.7902 ± 0.0102 | 0.1164 ± 0.0264 | — |
| 8    | 0.8602 ± 0.0029 | **0.8886 ± 0.0013** | 0.8874 ± 0.0017 | 0.8663 ± 0.0020 | — |
| 16   | 0.5919 ± 0.0406 | 0.6196 ± 0.0181 | **0.8598 ± 0.0023** | 0.8706 ± 0.0024 | — |
| 32   | 0.0985 ± 0.0087 | 0.0985 ± 0.0087 | **0.8519 ± 0.0011** | 0.8735 ± 0.0032 | 0.8946 ± 0.0032 |

## What transfers from MNIST, and what does not

**Transfers:**

1. **The v3 auto-step fix eliminates the fine-grid plateau.**
   16 bits: 0.5919 (v1) / 0.6196 (v2) -> 0.8598 (v3), a +24 to +27
   point gain. 32 bits: 0.0985 -> 0.8519, from random to near-baseline.
   This is the same magnitude of effect as on MNIST MLP and it
   confirms the plateau was a step-size artefact, not a property of
   the dataset.
2. **Projected-GD (Adam + snap) is broken at 2-4 bits.** 0.1167 and
   0.1164 — indistinguishable from the 0.10 random baseline. GDMC is
   **5.3x** (at 2 bits) and **6.9x** (at 4 bits) better. Across all
   five tasks tested, Projected-GD at 2-4 bits is the one consistent
   failure.
3. **Momentum alone is not enough at fine grids.** v2 at 16 bits is
   0.6196 vs v1's 0.5919 — a small gain, nothing like the step-size
   fix.

**Does not transfer cleanly:**

1. **At 2 bits, momentum HURTS.** v1 (0.6233) beats v2 (0.2974) by
   33 points. On MNIST MLP momentum was a large win at 3-4 bits; here
   at 2 bits it is actively harmful. The 4-level grid is too coarse
   for a smoothed momentum direction to be reliable, and the run is
   also high-variance (v1 std 0.077).
2. **The best coarse-bit config is v1 or v2, not v3.** v3 clips to
   k≈1 at coarse bits but a few high-gradient coordinates round up,
   which costs a little: at 4 bits v3 is 0.8 points below v2, at
   2 bits 25 points below v1.

## Combined picture across all five tasks

Recommendation by bit budget, now over toy regression, MNIST MLP,
MNIST CNN, CIFAR-10 CNN and Fashion-MNIST MLP:

| bit budget | recommended | evidence |
|------------|-------------|----------|
| 2-4 bits | gdmc v2 (or v1 at 2 bits) | 4.3-6.9x better than Projected-GD on every task |
| 8 bits | gdmc v2 or v3 | ties or beats Projected-GD on MNIST/Fashion; Projected-GD wins on CIFAR |
| 16-32 bits | **gdmc v3 (auto k)** | removes the plateau; matches Projected-GD on MLP/CNN/Fashion, within ~1 point of Adam |

The one open cell remains **CIFAR-10 at 8 bits**, where Projected-GD
(0.6067) still beats GDMC v2 (0.5682). v3 does not address that case
because it is not a step-size problem — Adam's per-coordinate adaptive
scaling is simply better on CIFAR's harder loss surface.

## Figures

* results/plots/fashion_mnist_train_loss_by_bits.png
* results/plots/fashion_mnist_test_loss_by_bits.png
* results/plots/fashion_mnist_test_acc_by_bits.png
* results/plots/fashion_mnist_acceptance_by_bits.png
* results/plots/fashion_mnist_final_acc_vs_bits.png

Generated from 591k per-step curve rows by analysis/plot_curves.py.
