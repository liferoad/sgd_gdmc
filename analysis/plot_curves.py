"""Comparison plots from per-step training curves.

Reads the per-run curve CSVs written by run_one(curves_dir=...) and
produces train/test loss and accuracy comparisons, plus acceptance-rate
and generalisation-gap views.

Usage::

    .venv/bin/python analysis/plot_curves.py \
        --curves-dir results/curves/mnist_mlp_v3 \
        --out-dir results/plots --prefix mnist_mlp_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.runner import read_curves


# A stable colour/marker per optimizer so figures are comparable.
STYLE = {
    "adam": ("#1f77b4", "o"),
    "momentum": ("#2ca02c", "s"),
    "sgd": ("#7f7f7f", "^"),
    "sgld": ("#bcbd22", "v"),
    "projected-gd": ("#ff7f0e", "D"),
    "gdmc": ("#d62728", "x"),
    "gdmc-adaptive": ("#9467bd", "P"),
    "gdmc-auto": ("#17becf", "*"),
}


def _style(label):
    for key, val in STYLE.items():
        if label.startswith(key):
            return val
    return ("#333333", ".")


def _variant(df):
    """A unique config key within an optimizer, from file name extras."""
    def key(row):
        parts = row["file"].split("__")
        extras = parts[5:] if len(parts) > 5 else []
        return row["optimizer"] + (" " + " ".join(extras) if extras else "")
    return df.apply(key, axis=1)


def _mean_over_seeds(df):
    """Average the curve over seeds, grouped by (variant, bits, step)."""
    df = df.copy()
    df["variant"] = _variant(df)
    g = df.groupby(["variant", "bits", "step"]).agg(
        loss=("loss", "mean"),
        loss_std=("loss", "std"),
        test_loss=("test_loss", "mean"),
        test_acc=("test_acc", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
    ).reset_index()
    return g


def _panels_by_bits(g, ycol, out_path, title, ylabel, logy=False,
                    variants=None):
    bits_vals = sorted(g["bits"].unique())
    n = len(bits_vals)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                             squeeze=False)
    var_list = variants if variants is not None else sorted(g["variant"].unique())
    for i, bits in enumerate(bits_vals):
        ax = axes[i // ncol][i % ncol]
        sub = g[g["bits"] == bits]
        for v in var_list:
            s = sub[sub["variant"] == v].sort_values("step")
            s = s[s[ycol].notna()]
            if s.empty:
                continue
            colour, marker = _style(v)
            ax.plot(s["step"], s[ycol], color=colour, lw=1.4,
                    marker=marker, markevery=max(1, len(s) // 12),
                    ms=3, label=v)
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves-dir", required=True)
    p.add_argument("--out-dir", default="results/plots")
    p.add_argument("--prefix", default="curves")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_curves(args.curves_dir)
    print(f"loaded {len(raw)} curve rows from {args.curves_dir}")
    print(f"  variants: {sorted(set(_variant(raw)))}")
    g = _mean_over_seeds(raw)

    _panels_by_bits(g, "loss", out / f"{args.prefix}_train_loss_by_bits.png",
                    "Training loss vs step (mean over seeds)",
                    "train loss (mini-batch)", logy=True)
    _panels_by_bits(g, "test_loss", out / f"{args.prefix}_test_loss_by_bits.png",
                    "Test loss vs step (mean over seeds)", "test loss", logy=True)
    _panels_by_bits(g, "test_acc", out / f"{args.prefix}_test_acc_by_bits.png",
                    "Test accuracy vs step (mean over seeds)", "test accuracy")
    _panels_by_bits(g, "acceptance_rate",
                    out / f"{args.prefix}_acceptance_by_bits.png",
                    "GDMC acceptance rate vs step (mean over seeds)",
                    "smoothed acceptance rate")

    # Final test accuracy vs bits, one line per variant.
    finals = []
    for (v, bits), sub in g.groupby(["variant", "bits"]):
        s = sub[sub["test_acc"].notna()].sort_values("step")
        if not s.empty:
            finals.append({"variant": v, "bits": bits,
                           "final_test_acc": s["test_acc"].iloc[-1],
                           "best_test_acc": s["test_acc"].max()})
    fin = pd.DataFrame(finals)
    if not fin.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for v in sorted(fin["variant"].unique()):
            s = fin[fin["variant"] == v].sort_values("bits")
            colour, marker = _style(v)
            ax.plot(s["bits"], s["final_test_acc"], color=colour, marker=marker,
                    lw=1.5, label=v)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("quantization bits")
        ax.set_ylabel("final test accuracy")
        ax.set_title("Final test accuracy vs bit-width")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{args.prefix}_final_acc_vs_bits.png", dpi=130)
        plt.close(fig)
        fin.to_csv(out / f"{args.prefix}_final_acc_vs_bits.csv", index=False)

    print(f"wrote plots to {out} with prefix {args.prefix}")


if __name__ == "__main__":
    main()
