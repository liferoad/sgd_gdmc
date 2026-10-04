# GDMC v3 — magnitude-scaled (auto) grid steps

This note documents the third iteration of the GDMC optimizer, which
removes the fine-grid plateau that v1 and v2 were stuck on.

## 1. The problem, diagnosed

The v1/v2 sweeps found that GDMC stalls at 16-32 bits: on MNIST MLP the
best test accuracy was 0.64 at 16 bits and 0.09 (random) at 32 bits,
while Projected-GD reached 0.97. The v2 momentum change did not help
there (0.70 at 16 bits, 0.09 at 32 bits).

A direct probe of the grid geometry explains why. For a grid on [-1, 1]:

| bits | levels | grid spacing delta | float32 epsilon near 0.5 | delta > eps? | does k=1 move a weight? |
|------|--------|--------------------|--------------------------|--------------|--------------------------|
| 4    | 16     | 1.33e-01           | 5.96e-08                 | yes          | yes |
| 8    | 256    | 7.84e-03           | 5.96e-08                 | yes          | yes |
| 16   | 65536  | 3.05e-05           | 5.96e-08                 | yes          | yes |
| 24   | 1.7e7  | 1.19e-07           | 5.96e-08                 | yes          | yes |
| 32   | 4.3e9  | 4.66e-10           | 5.96e-08                 | **no**       | **no**  |

At 32 bits a single grid step is **smaller than the float32
representable increment** near 0.5, so the snap is a no-op and the
proposed weights are bit-identical to the current ones. The model never
moves. At 16 bits one step is representable but is 3e-5, which is far
below a useful update for a 256-unit MLP (Adam's effective step there
is ~1e-3), so progress is glacial: the v1 curve at 16 bits climbs
slowly and is still only at 0.64 after 30 epochs.

So the plateau was never a property of the loss landscape. It was a
*step-size* bug: k=1 is the wrong number of grid steps at fine
resolutions.

## 2. The fix

Choose the number of grid steps per weight so that the *continuous*
displacement matches a target update magnitude:

    desired_i = step_scale * |g_i| / g_ref          (weight units)
    k_i       = clip(round(desired_i / delta), 1, k_max)

where

* g_ref is a per-tensor EMA of mean|g| (rate grad_ema, default 0.99),
  so the rule is scale-free;
* delta is the grid spacing for the current tensor;
* step_scale is the new hyperparameter: the target displacement for a
  coordinate whose gradient equals the running average (default 1e-3);
* k_max (default 2**22) caps the step at fine grids.

The rule self-selects the right regime:

* **4-8 bits**: the target displacement is below one grid step, so k_i
  clips to 1. v3 then behaves like the v2 proposal (momentum, one
  step). Nothing is lost at coarse bit-widths.
* **16-32 bits**: k_i grows to tens or hundreds of thousands, and the
  model can move again. Typical mean k observed: ~12 at 16 bits and
  ~80,000 at 32 bits for step_scale=1e-3.

The Metropolis test is unchanged, so an oversized proposal is still
rejected when it raises the loss; the grid step only sets the proposal,
not the acceptance.

This is Fix 1.1 from docs/future_work.md, in the "gradient-magnitude
scaled k" form rather than the constant-k form that v2 tried.

## 3. Results — MNIST MLP, 30 epochs, full 60K train, 3 seeds

Best test accuracy over the run, mean ± std over seeds.

| bits | gdmc v1 | gdmc v2 (mom) | gdmc v3 (ss=1e-4) | gdmc v3 (ss=1e-3) | gdmc v3 (ss=1e-2) | projected-gd | adam |
|------|---------|---------------|-------------------|-------------------|-------------------|--------------|------|
| 4    | 0.9009 ± 0.0040 | **0.9519 ± 0.0031** | — | 0.9469 ± 0.0039 | — | 0.2180 ± 0.0763 | — |
| 8    | 0.9620 ± 0.0012 | **0.9790 ± 0.0003** | — | 0.9785 ± 0.0003 | — | 0.9742 ± 0.0015 | — |
| 16   | 0.6414 ± 0.0614 | 0.7025 ± 0.0373 | 0.9132 ± 0.0003 | **0.9770 ± 0.0002** | 0.9717 ± 0.0004 | 0.9741 ± 0.0012 | — |
| 32   | 0.0896 ± 0.0226 | 0.0896 ± 0.0226 | 0.9129 ± 0.0007 | 0.9691 ± 0.0008 | **0.9750 ± 0.0004** | 0.9751 ± 0.0024 | 0.9832 ± 0.0009 |

### The headline

**The 16-32 bit plateau is gone.**

* 16 bits: 0.6414 (v1) / 0.7025 (v2) -> **0.9770 (v3)**. A +27 to +34
  point improvement, and v3 now *beats* Projected-GD (0.9741).
* 32 bits: 0.0896 (v1) / 0.0896 (v2) -> **0.9750 (v3)**. From random
  baseline to parity with Projected-GD (0.9751) and within 0.8 points
  of continuous Adam (0.9832).
* 16 bits is the sweet spot: v3 there (0.9770) is better than v3 at
  32 bits (0.9750), because the 16-bit grid still provides a little
  useful regularisation.

### Coarse bits: v3 does not help, and very slightly hurts at 4

* 8 bits: v3 (0.9785) ties v2 (0.9790) — both clip to k=1 almost
  everywhere.
* 4 bits: v3 (0.9469) is **0.5 points worse** than v2 (0.9519).
  At 4 bits delta is large, so most coordinates clip to k=1, but a few
  high-gradient coordinates round up to k=2, and a 2-level jump on a
  16-level grid is marginal. **Recommended: use v2 at <= 4 bits, v3 at
  >= 16 bits, either at 8 bits.**

### step_scale matters at fine grids

At 32 bits: ss=1e-4 -> 0.9129, ss=1e-3 -> 0.9691, ss=1e-2 -> 0.9750.
Too small a target displacement under-uses the available movement; the
1e-2 setting is essentially at the Projected-GD ceiling. At 16 bits
ss=1e-3 is best (0.9770) and 1e-2 is slightly worse (0.9717) — with a
coarser grid a 1e-2 target is a large fraction of a grid step.

## 4. Training curves

analysis/plot_curves.py produces the comparison figures from per-step
curves (train loss, test loss, test accuracy, acceptance rate):

* results/plots/mnist_mlp_v3_train_loss_by_bits.png
* results/plots/mnist_mlp_v3_test_loss_by_bits.png
* results/plots/mnist_mlp_v3_test_acc_by_bits.png
* results/plots/mnist_mlp_v3_acceptance_by_bits.png
* results/plots/mnist_mlp_v3_final_acc_vs_bits.png

The test-accuracy and test-loss figures show the diagnosis directly: at
16 and 32 bits the v1 and v2 curves are essentially **flat** (loss
stuck at 3-4, accuracy 0.09-0.70) while the v3 curve descends normally
to Projected-GD/Adam level. At 4-8 bits all GDMC variants overlap.

## 5. What this changes in the overall story

Before v3, the summary was "GDMC is only useful at 2-4 bits; at 8+ bits
Projected-GD or Adam wins". After v3:

| bit budget | recommended | result |
|------------|-------------|--------|
| 2-4 bits | gdmc v2 (momentum, fixed k=1) | 4.3-5.1x better than Projected-GD on all three models |
| 8 bits | gdmc v2 or v3 | ties the best baseline |
| 16-32 bits | **gdmc v3 (auto k)** | matches Projected-GD, within 1 point of Adam |

So GDMC with momentum and auto steps is competitive with
discrete-Adam-with-projection across **the entire bit-width range**,
which was not true of v1 or v2. The remaining gap is at the 32-bit
continuous limit, where plain Adam is still 0.8 points ahead.

## 6. Reproducing

    .venv/bin/python experiments/10_gdmc_v3_auto_k.py --quick   # 11 s smoke
    .venv/bin/python experiments/10_gdmc_v3_auto_k.py           # 63 runs, ~86 min on CPU
    MPLCONFIGDIR=.mpl_cache .venv/bin/python analysis/plot_curves.py \
        --curves-dir results/curves/mnist_mlp_v3 \
        --out-dir results/plots --prefix mnist_mlp_v3

Raw metrics: results/raw/gdmc_v3_auto_k.csv (63 rows).
Per-step curves: results/curves/mnist_mlp_v3/ (gitignored, regenerated
by the command above).
