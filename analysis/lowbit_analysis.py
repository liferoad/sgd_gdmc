"""Analyse the same-precision low-bit comparison.

Reads results/raw/lowbit_comparison*.csv and writes:

* results/lowbit_summary.md  - per (task, method, bits) mean / 95% CI and the
  paired differences vs projected Adam and vs FP32 Adam;
* results/lowbit_paired.csv  - the paired table in machine-readable form;
* results/plots/lowbit_learning_curves_<task>.png - train loss and test
  accuracy against step, one panel per bit-width.

Usage::

    .venv/bin/python analysis/lowbit_analysis.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.runner import read_curves

RAW_GLOB = "results/raw/lowbit_comparison*.csv"
METHOD_ORDER = ["adam-fp32", "qat-ste-adam", "gdmc-v2", "projected-adam",
                "gdmc-v1", "projected-msgd"]


def load_runs():
    frames = []
    for f in sorted(glob.glob(str(ROOT / RAW_GLOB))):
        if f.endswith("_quick.csv"):
            continue
        d = pd.read_csv(f)
        d["label"] = d["extra"].apply(lambda s: json.loads(s).get("label"))
        d["task_name"] = d["extra"].apply(
            lambda s: json.loads(s).get("task", "mnist"))
        frames.append(d)
    if not frames:
        raise SystemExit("no lowbit_comparison CSVs found")
    return pd.concat(frames, ignore_index=True)


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (x.mean() - h, x.mean() + h)


def summary_table(df, metric="best_test_acc"):
    rows = []
    for (task, label, bits), g in df.groupby(["task_name", "label", "bits"]):
        v = g[metric].values
        lo, hi = ci95(v)
        rows.append(dict(task=task, method=label, bits=bits, metric=metric,
                         n=len(v), mean=v.mean(),
                         std=v.std(ddof=1) if len(v) > 1 else np.nan,
                         ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def paired_table(df):
    """Paired differences for both best-of-run and final accuracy.

    Reporting both matters: projected Adam peaks early and then collapses at
    2-4 bits, so best-of-run flatters it and final accuracy exposes it.
    """
    rows = []
    for metric in ("best_test_acc", "final_test_acc"):
        for task, gt in df.groupby("task_name"):
            # FP32 Adam is bit-independent: look it up per task, not per bits.
            ref = gt[gt.label == "adam-fp32"].set_index("seed")[metric]
            for bits, g in gt.groupby("bits"):
                base = g[g.label == "projected-adam"].set_index("seed")[metric]
                for label in METHOD_ORDER:
                    s = g[g.label == label].set_index("seed")[metric]
                    if s.empty:
                        continue
                    common = sorted(set(s.index) & set(base.index))
                    if label != "projected-adam" and len(common) > 1:
                        d = (s[common] - base[common]).values
                        t = stats.ttest_rel(s[common].values, base[common].values)
                        try:
                            w = stats.wilcoxon(s[common].values, base[common].values)
                            wp = float(w.pvalue)
                        except ValueError:
                            wp = float("nan")
                        rows.append(dict(task=task, bits=bits, method=label,
                                         metric=metric,
                                         comparison="vs projected-adam",
                                         n=len(common), mean_diff=d.mean(),
                                         median_diff=float(np.median(d)),
                                         paired_p=float(t.pvalue), wilcoxon_p=wp,
                                         n_positive=int((d > 0).sum())))
                    common = sorted(set(s.index) & set(ref.index))
                    if label != "adam-fp32" and len(common) > 1:
                        d = (s[common] - ref[common]).values
                        t = stats.ttest_rel(s[common].values, ref[common].values)
                        rows.append(dict(task=task, bits=bits, method=label,
                                         metric=metric,
                                         comparison="vs FP32 Adam",
                                         n=len(common), mean_diff=d.mean(),
                                         median_diff=float(np.median(d)),
                                         paired_p=float(t.pvalue),
                                         wilcoxon_p=float("nan"),
                                         n_positive=int((d > 0).sum())))
    return pd.DataFrame(rows)


def plot_curves(task, out_path):
    cdir = ROOT / "results" / "curves" / f"lowbit_{task}"
    if not cdir.exists():
        return False
    d = read_curves(cdir)
    d["method"] = d["optimizer"]
    # QAT shares the "adam" optimizer name but uses a grid.
    d.loc[(d.method == "adam") & (d.grid_spec == "uniform"), "method"] = "qat-ste-adam"
    d.loc[(d.method == "adam") & (d.grid_spec == "none"), "method"] = "adam-fp32"
    d.loc[(d.method == "gdmc") & (d.get("beta1", 0) > 0), "method"] = "gdmc-v2"
    d.loc[(d.method == "gdmc") & (~(d.get("beta1", 0) > 0)), "method"] = "gdmc-v1"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    colors = {"gdmc-v1": "tab:blue", "gdmc-v2": "tab:orange",
              "projected-adam": "tab:green", "qat-ste-adam": "tab:red",
              "adam-fp32": "k"}
    for j, bits in enumerate((2, 4, 8)):
        sub_bits = d[d.bits == bits]
        ref = d[(d.method == "adam-fp32")]
        for method in ["gdmc-v1", "gdmc-v2", "projected-adam", "qat-ste-adam"]:
            m = sub_bits[sub_bits.method == method]
            if m.empty:
                continue
            for i, col in enumerate(["loss", "test_acc"]):
                ax = axes[i, j]
                series = m.dropna(subset=[col])
                if series.empty:
                    continue
                g = series.groupby("step")[col].mean()
                ax.plot(g.index.values, g.values, "-", color=colors[method],
                        label=method, lw=1.2 if col == "test_acc" else 1.0,
                        alpha=0.9)
        for i, col in enumerate(["loss", "test_acc"]):
            ax = axes[i, j]
            if col == "test_acc" and not ref.empty:
                pts = ref.dropna(subset=["test_acc"])
                g = pts.groupby("step")["test_acc"].mean()
                ax.plot(g.index, g.values, "--", color="k", lw=1.2,
                        label="adam-fp32")
            ax.set_title(f"{bits}-bit" if i == 0 else "")
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("train loss" if i == 0 else "test accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{task}: GDMC vs same-precision baselines (mean over seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    df = load_runs()
    summ = pd.concat([summary_table(df, "best_test_acc"),
                      summary_table(df, "final_test_acc")], ignore_index=True)
    paired = paired_table(df)

    md = ["# Same-precision low-bit comparison - summary\n",
          f"Runs: {len(df)} from {df.task_name.nunique()} task(s) and "
          f"{df.seed.nunique()} distinct seeds. Settings were selected on a "
          "held-out validation split per method and bit-width (see "
          "results/lowbit_settings_*.json).\n",
          "## Mean test accuracy (95% t-interval over seeds)\n",
          "best_test_acc is the peak over the run; final_test_acc is the last",
          "epoch. Reporting both matters because projected Adam peaks early and",
          "then collapses at 2-4 bits.\n",
          summ.sort_values(["metric", "task", "bits", "method"]).to_markdown(
              index=False, floatfmt=".4f"),
          "",
          "## Paired differences (matched by seed)\n",
          paired.sort_values(["metric", "task", "bits", "comparison",
                              "method"]).to_markdown(index=False,
                                                     floatfmt=".4f"),
          ""]
    for task in sorted(df.task_name.unique()):
        path = ROOT / "results" / "plots" / f"lowbit_learning_curves_{task}.png"
        if plot_curves(task, path):
            md.append(f"Learning curves: results/plots/{path.name}\n")
    (ROOT / "results" / "lowbit_summary.md").write_text("\n".join(md))
    paired.to_csv(ROOT / "results" / "lowbit_paired.csv", index=False)
    summ.to_csv(ROOT / "results" / "lowbit_summary.csv", index=False)
    print(f"Wrote results/lowbit_summary.md ({len(summ)} cells, "
          f"{len(paired)} paired rows)")


if __name__ == "__main__":
    main()
