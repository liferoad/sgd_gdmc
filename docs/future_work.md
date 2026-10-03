# Future work — improving GDMC for deep-learning weight optimization

This document is a brain-dump of concrete ideas for making the GDMC
optimizer a stronger baseline. It is organized by the limitations
listed in [results/REPORT.md](../results/REPORT.md) and adds a few
extra directions I noticed while thinking about the data.

Notation used below:
* `g` — gradient of the loss w.r.t. weights
* `delta` — current grid spacing in `[vmin, vmax]`
* `m_t`, `v_t` — first / second moment of `g` (Adam-style)
* `k` — number of *grid steps* to take at a coordinate
* `β` — Metropolis temperature (existing)
* `α` — `m_t`-based step-size scaler (new in some ideas)

---

## L1. One grid step is suboptimal at fine grids

The biggest observed failure: at 16/32 bits, the "one grid step in
`-sign(g)` direction" is essentially a no-op (a 32-bit step on a
[-1, 1] range is `2/2^32 ≈ 5e-10`). Acceptance stays at 1.0 because
nothing changes, but nothing improves either.

### Fix 1.1 — Multi-step moves (step `k` grid points at a time)

Replace `sign(g)` with `k · sign(g)` for some `k ≥ 1`. Two ways to
choose `k`:

* **Fixed k** (e.g. 1, 2, 4, 8). Trivial to implement, easy to ablate.
* **Gradient-magnitude scaling** —
  `k_i = max(1, round(alpha * |g_i| / grad_scale))` where `grad_scale`
  is a per-tensor EMA of `|g|`. The intuition: a small `|g|` means the
  loss surface is flat in that coordinate, so step lightly; a large
  `|g|` means we're far from a minimum, so step further.

`k` is then clipped to `[1, k_max]`, e.g. `k_max = 32` for 5-bit grids
or `min(2^(bits-1), 64)` as a heuristic.

**Expected effect**: 16/32-bit GDMC stops being stuck. Acceptance
rate will drop (more drastic moves are more often rejected) but the
accepted moves cover more ground, so the *effective* step size
(`acceptance_rate × k × delta`) becomes non-trivially positive at all
bit-widths.

**Evidence the fix works**: GDMC at 16/32 bits on MNIST MLP should
climb from the current ~9% accuracy into the 80%+ range, matching
Projected-GD.

**Risk**: too-large `k` can skip over good grid points. Cap and
ablate.

### Fix 1.2 — Move-set drawn from high-`|g|` weights

At fine grids, the bottleneck is *which* weights get moved. Replace
the uniform random move-set with a per-tensor top-`|g|` selection
(probabilistically — Gumbel-top-k or just `bernoulli(|g|/|g|_max)`).
This is the "importance sampling" analog of the move-set.

**Expected effect**: at 16/32 bits, every move is on a weight that
*actually* matters, so we waste fewer steps on near-zero-gradient
weights.

**Risk**: requires a per-tensor pass over `|g|`, not free.

---

## L2. Memoryless proposal — no momentum

The current GDMC is `proposal = -sign(g)`. There is no analog of
Adam's `m_t`, so on noisy minibatches the proposal direction jitters
and the algorithm does a random walk at coarse grids where most
moves are rejected.

### Fix 2.1 — First-moment momentum

Maintain `m_t = β1 * m_{t-1} + (1 - β1) * g_t` and use `sign(m_t)` as
the proposal direction. This is a discrete Adam.

Defaults: `β1 = 0.9` (same as Adam), warm-up the bias by dividing by
`1 - β1^t` on the first few steps.

**Expected effect**: smoother proposals, fewer wasted moves at coarse
grids, faster convergence at fine grids. This is the single change I
expect to give the largest absolute gain.

**Evidence**: Adam's success over SGD on the same architectures is
largely about the first moment. The discrete analog should inherit
most of the benefit.

### Fix 2.2 — Per-tensor second moment (full Adam)

Add `v_t = β2 * v_{t-1} + (1 - β2) * g_t^2` and use
`m_t / (sqrt(v_t) + eps)` as the *direction* (`sign` of this is the
proposal) and a per-tensor scaling of `k` (Fix 1.1's
gradient-magnitude form). This is a fully-discrete Adam.

**Expected effect**: scale-invariant per-tensor step sizes — large
gradient *variance* → small `k`; small variance → larger `k`. Should
make a single `β` work across all layers (BN has small gradients,
classifier FC has large).

**Risk**: two more hyperparameters to tune.

---

## L3. Single β for everything

Currently `β` is a global scalar. Different tensors (BN vs FC) have
very different loss scales. A bad `β` for one tensor ruins the
search for the other.

### Fix 3.1 — Per-tensor β

`β_tensor = β_global / (per-tensor dL std)`. Compute the std of
`dL = L_new - L_old` per tensor over a short rolling window, and
normalize the per-tensor `β` so the acceptance probability is
approximately uniform across tensors.

**Expected effect**: BN layers stop dragging the search; the
classifier FC stops dominating. Acceptance rate should look more
uniform across the network.

### Fix 3.2 — Adaptive global β

Start with `β = β_high` (eager, near-greedy) and anneal to
`β = β_low` (relaxed, exploratory) over training. This is
"simulated annealing in β" — borrow the spirit of SGLD's noise
decay for the temperature schedule.

**Expected effect**: early steps accept less but commit to a
direction; later steps can escape the local minimum more easily.

**Risk**: needs a schedule (linear, cosine, or step).

---

## L4. The fully adaptive grid was retired — bring it back with EMA

`AdaptiveGrid` is in the code but not used because the per-step
`vmax = max|w|` re-derivation caused runaway weight growth. The
fix is the standard trick: smooth `vmax` with an exponential moving
average.

### Fix 4.1 — Slow-EMA adaptive grid

```
vmax_t = max(alpha, EMA_beta * vmax_{t-1} + (1 - EMA_beta) * max|w_t|)
vmin_t = -vmax_t
```

Pick `EMA_beta = 0.99` so the grid range adjusts over ~100 steps
rather than per step. Set `alpha = 1.05` to leave headroom for weights
that grow.

**Expected effect**: the grid tracks the weight magnitude without
locking in the runaway. Compared to `uniform-pt` (which always uses
the current `max|w|` at the moment of evaluation), the EMA version
is stable across long training runs and gives the optimizer a
consistent discretization target.

---

## L5. Short training horizon

2–3 epochs isn't enough to see GDMC's full advantage at coarse
grids. The paper used thousands of iterations per sequence.

### Fix 5.1 — Long-horizon runs (engineering, not algorithmic)

Run the headline sweep at 10–20 epochs (or until the test loss
plateaus). Trivially 5–10x more compute; should be done before any
algorithmic changes are evaluated so the comparison is fair.

### Fix 5.2 — Coarse-to-fine progressive quantization

Initialize at 4 bits, train until convergence, then snap to 8 bits
and continue training, then 16, then 32. Each snap is a small
perturbation of a converged model, so each stage trains quickly.

**Expected effect**: sidesteps the cold-start problem at fine grids
(idea L1) by always starting the next bit-width from a converged
solution at a coarser bit-width. This is well-known in QAT
literature as "incremental quantization".

**Risk**: at each transition the loss jumps (snap is a perturbation).
Tolerable, but a small `lr` warm-down at the transition helps.

### Fix 5.3 — Warm-start from a continuous Adam baseline

Train a high-precision model with Adam, then snap to the grid and
fine-tune with GDMC. This is "post-training quantization with GDMC
fine-tuning" and is probably the most practically valuable use
case for GDMC in deployment.

**Expected effect**: GDMC's role is then *correction* of
quantization error on a strong starting point. Acceptance rate
should be much higher (proposed moves are small refinements) and
test accuracy should be close to or better than naive snap-to-grid.

---

## L6. Minibatch-noise bias in the acceptance test

The acceptance test uses the *same* minibatch whose gradient defined
the move. This is biased toward acceptance (the move was constructed
to decrease that loss).

### Fix 6.1 — Held-out Metropolis batch

Split each minibatch into two halves: `B_grad` for the gradient
(proposal) and `B_accept` for the acceptance test. With `B` of 128
that's 64 + 64. Or keep a fixed 1000-example held-out batch
rotated every K steps.

**Expected effect**: more honest acceptance rate, less random walk.

**Risk**: roughly 2x compute per step (one extra forward pass).

---

## L7. Projected-GD vs GDMC at ≥ 8 bits — closing the gap

Projected-GD with Adam recovers at 8+ bits. The natural follow-up:
**GDMC-with-Adam-proposal**, where the proposal is not just
`sign(g)` but the *Adam-updated* weights projected to the grid.

### Fix 7.1 — Adam-direction proposal

```
m_t, v_t = adam_update(g_t)
proposed = snap(w_t - lr * m_t / (sqrt(v_t) + eps))   # grid-snapped Adam step
```

The acceptance test is unchanged. If the move is accepted, the
parameters are updated to `proposed`; if rejected, restored to the
previous (snapped) `w_t`.

**Expected effect**: at fine grids this is "Adam with snap".
At coarse grids the acceptance criterion thwarts Adam's worst
instincts (overshooting on the discrete surface). Should beat both
plain Adam and plain Projected-GD across all bit-widths.

**Risk**: degrades to Projected-GD if acceptance → 1 always. Need to
ablate `lr` carefully.

---

## L8. Smarter grid shapes

The current grid is uniform in `[vmin, vmax]`. Trained neural
networks have heavy-tailed weight distributions; a uniform grid
wastes bits on the rare large weights.

### Fix 8.1 — Log-spaced grid

Levels at `±delta, ±2*delta, ±4*delta, …` (powers of 2 around 0)
plus a few extra levels near 0 for fine resolution. E.g. 4-bit
log-grid in `[-1, 1]`:
`{0, ±0.06, ±0.125, ±0.25, ±0.5, ±1}`. 16 levels, more density
near 0.

**Expected effect**: better quantization for the same bit count,
particularly at coarse bits.

**Risk**: the "neighbour" step is no longer constant; need to use
*log-ratio* step in the proposal. Slightly more code.

### Fix 8.2 — Learned codebook (k-means)

Cluster the weights every N steps; use the cluster centers as the
grid. K = 16 for 4-bit. This is "learned quantization" applied to
GDMC.

**Expected effect**: best quantization quality, but the move-set
and neighbour definitions change. Significant implementation work.

---

## L9. Activation quantization (for an honest deployment story)

Right now we only quantize *weights*. The natural next step is also
quantizing activations to fake-quantize through the forward pass at
training time, so the trained model matches what would actually run
on a quantized device.

**Expected effect**: more realistic test numbers; the optimizer
learns to be robust to activation noise, which is the dominant
noise source in modern QAT.

**Risk**: doubles memory and compute for the forward pass.
Straightforward to add with fake-quantization modules around each
activation.

---

## L10. Better baselines (so the comparison is fair)

The current SGLD baseline has no Metropolis acceptance — it's a
"stochastic gradient Langevin *without* accept/reject". A more
honest comparison is **SG-MCMC** (Metropolis-Hastings SGLD). It
exists, it's the natural MCMC cousin of GDMC, and it would clarify
how much of GDMC's win comes from the *discrete move* (vs the
accept/reject alone).

**Expected effect**: better attribution of where GDMC's gains come
from.

**Risk**: needs a good reference implementation. The SGLD+
Metropolis version is in Ma et al. 2015 ("A Complete Recipe for
Stochastic Gradient MCMC") and is straightforward to add.

---

## L11. Faster proposal construction (engineering)

GDMC is ~2x more expensive per step than Adam because of the second
forward pass for the acceptance test. The two cheapest wins:

* **Acceptance on the same minibatch, but reuse the autograd graph**
  — record the loss for both old and proposed weights in one
  forward pass by using `torch.func.functional_call` with two
  parameter sets. This saves the second forward entirely.
* **Snap on the fly** — currently we materialize the full snapped
  tensor. For `move_frac = 0.01` we only need 1% of indices; do
  the snap lazily.

Both are independent of the algorithm and drop straight in.

---

## L12. Diagnostics that are missing

The runner currently records `final_acceptance_rate` and
`wall_time_sec`. Two more cheap-but-useful columns:

* **Quantization error** `||w_continuous - w_quantized||_2 / ||w_continuous||_2`
  at the end of training. Easy to add in the runner.
* **Effective step size** `mean(k_i * delta_i * I[accepted])`
  over a rolling window. Captures the *real* progress per step,
  not just the proposal magnitude.

These would make the loss-curves plots much more interpretable.

---

## Priority list — what to try first

If we have ~1 week of compute, the highest-impact / lowest-risk
changes are:

1. **Fix 2.1** (momentum) — directly addresses L2. Highest expected
   absolute gain. Easy to implement (10 lines in `_make_step`).
2. **Fix 1.1** (multi-step moves) — directly addresses L1.
   Single-line change, sweep `k ∈ {1, 2, 4, 8, 16}`.
3. **Fix 5.3** (warm-start from Adam) — reframes the question from
   "can GDMC train from scratch?" (no, especially at fine grids)
   to "can GDMC fine-tune a quantized model?" (yes, often the
   practical question).
4. **Fix 4.1** (slow-EMA adaptive grid) — re-enables the truly
   adaptive grid that was retired. Stable.
5. **Fix 6.1** (held-out Metropolis batch) — improves honesty of
   the acceptance rate. Modest compute cost.

Then, in roughly this order:
6. Fix 3.2 (β schedule) — simple, helps a bit.
7. Fix 7.1 (Adam-direction proposal) — biggest open question
   ("can GDMC and Adam coexist?").
8. Fix 5.2 (coarse-to-fine) — nice for long training.
9. Fix 8.1 (log-spaced grid) — algorithmic, more code.
10. Fix 9 (activation quantization) — engineering, big surface
    area.

The **GDMC v2** I'd want to write first: Fixes 2.1 + 1.1 + 4.1 +
6.1. About 50 lines of changes to `optimizer.py` and `grid.py`,
plus a new option to `run_one` to record the quantization error.

---

## What this *won't* fix

* GDMC will not beat Adam at the 32-bit continuous limit — the
  step quantization always costs you something, and Adam is
  already near-optimal on smooth loss surfaces.
* GDMC is not a substitute for proper QAT frameworks that quantize
  both weights and activations. It's a research prototype to study
  the *discrete move* idea, not a deployment tool.
* The fundamental limit of "one grid step per move" at fine grids is
  computational (we'd have to evaluate the loss thousands of times
  per step to do a real Adam step on a 32-bit grid). Fix 1.1 helps but
  is not free.

---

## Status of the ideas (as of GDMC v2)

The first two ideas on the priority list have been implemented and
tested. The empirical results, in [v2_results.md](v2_results.md),
are partly encouraging and partly sobering:

### Fix 2.1 (momentum, `beta1=0.9`) — **shipped, big win at 3-4 bits**

* At 3-bit quantization: test loss `0.398 → 0.147` (2.7× improvement).
* At 4-bit quantization: test loss `0.367 → 0.101` (3.6× improvement).
* At 8+ bits: matches v1.
* At 2 bits: no change (still broken).

This is the biggest algorithmic improvement in the project. The
momentum buffer smooths out the gradient noise on coarse grids,
which is exactly the regime where GDMC is uniquely valuable.

### Fix 1.1 (multi-step moves, constant `k`) — **shipped, mixed**

* At 8-bit: best `k=4` gives `0.099 → 0.029` (3.4× improvement over v1).
* At 3-4 bits: HURTS. Acceptance rate collapses (k=4 at 3-bit is a
  jump of half the grid range; most moves are rejected).
* At 16+ bits: marginal — even `k=4` is too small relative to the
  grid resolution.

So **constant-k multi-step is only useful at 8 bits**. The
practical recommendation: at 4-bit use `k=1`; at 8-bit use `k=4`;
at 16+ bits, the v2 plateau remains.

### What was learned: the *real* Fix 1.1 should be magnitude-scaled

Constant `k` is the wrong shape. The paper's intuition (line-search
along `-sign(g)`) implies `k` should be *proportional to* `|g|`:
large `|g|` (we're far from a minimum) → big `k`; small `|g|` (flat
direction) → small `k` (= 1). This unifies the three zones:

* 3-4 bits: gradient is mostly noise, so `|g|` is small; `k=1`
  dominates. **But momentum smooths `|g|`**, which is why it helps
  here.
* 8 bits: gradient is meaningful, `|g|` is moderate, so `k≈4`
  dominates. Matches what we see.
* 16+ bits: `|g|` is small (loss is flat), `k≈1` dominates. Same
  plateau as v1.

The next experiment should be: `k = clip(round(k_target * |g| /
grad_scale_EMA), 1, k_max)` with `k_target` and `k_max` as new
hyperparameters. Expected: 2.7-3.6× win at 3-4 bits (carried over
from momentum) *and* a continued win at 8+ bits (where the
magnitude-scaled k is much larger than constant-4 for the
high-`|g|` coordinates). This is the **GDMC v3** target.

