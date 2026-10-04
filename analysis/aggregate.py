"""Aggregate all experiment CSVs into headline tables and plots.

Reads every CSV in ``results/raw/``, computes mean ± std across seeds for
each (task, optimizer, bits, grid) combination, and writes:

* ``results/headline_table.md`` — markdown table of best test accuracy
  per (task, optimizer, bits) with mean ± std across seeds.
* ``results/headline_table.csv`` — same data in CSV form.
* ``results/plots/<task>_loss.png`` — per-task loss curves (one curve per
  optimizer at each bits level).
* ``results/plots/<task>_acceptance.png`` — for GDMC runs only,
  acceptance rate over training.
* ``results/REPORT.md`` — the human-readable final report.

Usage::

    .venv/bin/python analysis/aggregate.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt


def _read_all(raw_dir):
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {raw_dir}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        df["__source__"] = f.stem
        frames.append(df)
    if not frames:
        raise SystemExit("no non-empty CSVs in " + str(raw_dir))
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df


def headline_table(df, metric):
    """Group by (task, model, optimizer, bits) and compute mean ± std of metric."""
    g = df.groupby(["task", "model", "optimizer", "grid_spec", "bits"])[metric].agg(
        ["mean", "std", "count", "min", "max"]
    ).reset_index()
    g["mean_std"] = g.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if r["count"] > 1 else f"{r['mean']:.4f}",
        axis=1,
    )
    return g


def _metric_for(task):
    """Return the headline metric name for a task.

    Classification tasks: best test accuracy (higher is better).
    Regression tasks: best test loss (lower is better).
    """
    if task == "toy_regression" or task == "toy_regression_v2":
        return "best_test_loss"
    return "best_test_acc"


def _fmt_table(df, metric):
    cols = ["task", "model", "optimizer", "grid_spec", "bits", "mean_std", "min", "max", "count"]
    out = df[cols].to_markdown(index=False, floatfmt=".4f")
    if metric == "best_test_loss":
        out = out.replace("mean_std", "mean_std (test_loss)")
    else:
        out = out.replace("mean_std", "mean_std (test_acc)")
    return out


def plot_loss_curves(df, task, out_path):
    sub = df[df["task"] == task]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (opt, grid), g in sub.groupby(["optimizer", "grid_spec"]):
        g = g.sort_values("bits")
        if g["bits"].nunique() < 2:
            continue
        agg = g.groupby("bits")["best_test_acc"].mean().reset_index()
        label = f"{opt}/{grid}"
        ax.plot(agg["bits"], agg["best_test_acc"], "o-", label=label, alpha=0.8)
    ax.set_xlabel("Quantization bits")
    ax.set_ylabel("Best test accuracy")
    ax.set_title(f"{task}: best test accuracy vs quantization bits")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_acceptance(df, task, out_path):
    sub = df[(df["task"] == task) & (df["optimizer"].str.startswith("gdmc"))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for (opt, grid), g in sub.groupby(["optimizer", "grid_spec"]):
        g = g.sort_values("bits")
        agg = g.groupby("bits")["final_acceptance_rate"].mean().reset_index()
        label = f"{opt}/{grid}"
        ax.plot(agg["bits"], agg["final_acceptance_rate"], "o-", label=label, alpha=0.8)
    ax.set_xlabel("Quantization bits")
    ax.set_ylabel("Final acceptance rate (smoothed)")
    ax.set_title(f"{task}: GDMC acceptance rate vs quantization bits")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


LONG_RUN_TASKS = {"mnist_mlp_long", "mnist_cnn_long", "mnist_cnn_long_focused",
                  "toy_regression_v2"}


def write_report(df, out_path):
    lines = []
    lines.append("# GDMC for Deep Learning Weight Optimization — Final Report\n")
    lines.append("This report aggregates the *short* sweep in `results/raw/*.csv`.\n")
    lines.append("## Top-line conclusion\n")
    lines.append("There is no universal-best optimizer; the right choice depends on the bit-width budget.\n")
    lines.append("**The long runs overturn the short-run picture at 8 bits.** A 30-epoch re-run on MNIST MLP (full 60K training set) gives:\n")
    lines.append("| optimizer | bits | best test acc |")
    lines.append("|---|---|---|")
    lines.append("| adam (continuous) | 32 | 0.9757 ± 0.0014 |")
    lines.append("| **gdmc v2 (β1=0.9, k=1)** | **8** | **0.9775 ± 0.0005** |")
    lines.append("| gdmc v2 (β1=0.9, k=1) | 4 | 0.9460 ± 0.0017 |")
    lines.append("| gdmc v1 | 8 | 0.9608 ± 0.0004 |")
    lines.append("| gdmc v1 | 4 | 0.8959 ± 0.0021 |")
    lines.append("| projected-gd | 4 | 0.2180 ± 0.0763 |")
    lines.append("")
    lines.append("**GDMC v2 at 8-bit matches/beats continuous Adam** (0.9775 vs 0.9757), and at 4-bit it is 3.0 points behind Adam while being 4.3× better than the QAT-style Projected-GD baseline. The short sweep below used only 2-3 epochs and understates GDMC substantially (GDMC 8-bit gains +0.186 from 3→30 epochs, the largest of any config). Full long-run analysis in `docs/long_runs.md`; per-cell best-optimizer tally in `docs/test_report.md` §3.7; v2 sweep in `docs/v2_results.md`.\n")

    # Long-run table (so the auto-generated report also surfaces the
    # long-run conclusion).
    long_run_table = [
        ("", ""),
        ("## Top-line conclusion from the long runs (see `docs/long_runs.md` for the full analysis)\n", ""),
        ("The same 3-model × (Adam 32-bit / GDMC v2 8-bit / GDMC v2 4-bit / Projected-GD 4-bit) table, now at a proper training horizon:\n", ""),
        ("| model | long run | Adam (32-bit) | GDMC v2 @ 8-bit | GDMC v2 @ 4-bit | Projected-GD @ 4-bit |\n", ""),
        ("|---|---|---|---|---|---|", ""),
        ("| MNIST MLP | 30 epochs, 60K train | 0.9757 | **0.9775** | 0.9460 | 0.2180 |\n", ""),
        ("| MNIST CNN | 8 epochs, 10K train  | 0.9546 | **0.9668** | 0.7807 | 0.1503 |\n", ""),
        ("| CIFAR-10 CNN | 20 epochs, 10K train | 0.5747 | 0.5682 | 0.5538 | 0.1217 |\n", ""),
        ("GDMC v2 with momentum matches or beats continuous Adam at 8-bit on MNIST MLP and CNN, and is **4.3-5.1× better than Projected-GD at 4-bit on all three models**. On CIFAR-10 at 8-bit the order inverts (Projected-GD 0.6067 > Adam 0.5747 > GDMC v2 0.5682), but GDMC v2 still beats Projected-GD at 4-bit on CIFAR by 4.55×.\n", ""),
    ]
    for line, _ in long_run_table:
        lines.append(line)

    lines.append("## Headline table — best metric by (task, optimizer, bits)\n")
    lines.append("*(short sweep only; long-run tasks are excluded and reported in `docs/long_runs.md`)*\n")
    for task in sorted(df["task"].unique()):
        if task == "beta_sweep" or task in LONG_RUN_TASKS:
            continue
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        metric = _metric_for(task)
        tab = headline_table(sub, metric)
        lines.append(f"### {task}  (metric: {metric})\n")
        lines.append(_fmt_table(tab, metric) + "\n")
        lines.append("")
    if any("beta_sweep" in s for s in df["__source__"]):
        bs = df[df["__source__"].str.contains("beta_sweep", na=False)]
        if not bs.empty:
            lines.append("## Beta sweep (MNIST CNN, 4-bit)\n")
            for src, g in bs.groupby("__source__"):
                lines.append(f"### {src}\n")
                lines.append(_fmt_table(headline_table(g, "best_test_acc"), "best_test_acc") + "\n")
                lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="results/raw")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _read_all(raw_dir)
    print(f"Loaded {len(df)} rows from {raw_dir}.")
    print(f"Tasks: {sorted(df['task'].unique())}")
    print(f"Optimizers: {sorted(df['optimizer'].unique())}")
    print(f"Bits: {sorted(df['bits'].unique())}")

    short_df = df[~df["task"].isin(LONG_RUN_TASKS)]
    tab = headline_table(short_df, "best_test_acc")
    tab.to_csv(out_dir / "headline_table.csv", index=False)
    md = ["# Headline table — best test accuracy by (task, optimizer, bits)\n",
          "mean ± std over seeds. count = number of seeds.\n",
          "Short sweep only; long-run tasks are reported in `docs/long_runs.md`.\n",
          _fmt_table(tab, "best_test_acc") + "\n"]
    (out_dir / "headline_table.md").write_text("\n".join(md))

    for task in sorted(df["task"].unique()):
        if task == "beta_sweep":
            continue
        plot_loss_curves(df, task, plots_dir / f"{task}_loss.png")
        plot_acceptance(df, task, plots_dir / f"{task}_acceptance.png")

    write_report(df, out_dir / "REPORT.md")
    print(f"Wrote headline table, plots, and REPORT.md to {out_dir}")


if __name__ == "__main__":
    main()
