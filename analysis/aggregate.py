"""Aggregate all experiment CSVs into headline tables and plots.

Reads every CSV in results/raw/, computes mean / std / 95% CI across seeds
for each **full** configuration (including beta1, k, step_scale, noise and
acceptance settings), and writes:

* results/headline_table.md   - grouped table with CIs
* results/headline_table.csv  - same data in CSV form
* results/plots/<task>_*.png  - loss / acceptance curves
* results/REPORT.md           - report generated entirely from the data

Two bugs fixed relative to the original aggregator:

1. It grouped only by (task, model, optimizer, grid_spec, bits), silently
   pooling different beta1 / k / step_scale configurations into one row.
   The group key now includes every experimental setting (including beta, lr,
   epochs, batch size, acceptance and noise), and count is the number of
   distinct seeds.
2. Unquantized baselines were replicated at every bits value in the long
   runs; those copies are de-duplicated, but only against runs identical in
   every other setting. The JSON extra column is expanded BEFORE
   de-duplication, so runs that differ only in gradient-noise level are never
   collapsed.

Usage::

    .venv/bin/python analysis/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import math
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


# Every setting that defines a run. Two rows may only be pooled if all of
# these match.
CONFIG_KEYS = ["task", "model", "optimizer", "grid_spec", "bits",
               "epochs", "batch_size", "lr", "momentum", "beta", "move_frac",
               "beta1", "k", "k_mode", "step_scale", "project_lr_scale",
               "grad_noise_rho", "accept_on", "select_mode"]


def _read_all(raw_dir):
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {raw_dir}")
    # Only per-run result CSVs participate; the directory also holds
    # benchmarks with a different schema (e.g. memory_benchmark.csv).
    required = {"task", "optimizer", "bits", "seed"}
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if not required.issubset(df.columns):
            continue
        df["__source__"] = f.stem
        frames.append(df)
    if not frames:
        raise SystemExit("no non-empty CSVs in " + str(raw_dir))
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Expand first: de-duplication must be able to see the settings that live
    # in the JSON extra column (noise level, k_mode, step_scale, ...).
    return _dedup_unquantized(_expand_extra(df))


def _expand_extra(df):
    """Promote the JSON extra column into flat hyperparameter columns."""
    for col, default in [("k_mode", "fixed"), ("step_scale", np.nan),
                         ("project_lr_scale", np.nan), ("grad_noise_rho", 0.0),
                         ("accept_on", "minibatch"),
                         # Runs written before select_mode existed used the
                         # legacy full-permutation path.
                         ("select_mode", "randperm"),
                         ("beta1", 0.0), ("k", 1)]:
        if col not in df.columns:
            df[col] = default
    if "extra" in df.columns:
        parsed = df["extra"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {})
        for col in ["k_mode", "step_scale", "project_lr_scale",
                    "grad_noise_rho", "accept_on", "select_mode"]:
            df[col] = [p.get(col, d) if isinstance(p, dict) else d
                       for p, d in zip(parsed, df[col])]
    df["beta1"] = pd.to_numeric(df["beta1"], errors="coerce").fillna(0.0)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(1).astype(int)
    df["k_mode"] = df["k_mode"].fillna("fixed")
    df["accept_on"] = df["accept_on"].fillna("minibatch")
    df["grad_noise_rho"] = pd.to_numeric(
        df["grad_noise_rho"], errors="coerce").fillna(0.0)
    return df


def _dedup_unquantized(df):
    """Unquantized baselines have no meaningful bits; keep one row per run."""
    if "grid_spec" not in df.columns:
        return df
    unq = df["grid_spec"].astype(str).str.lower() == "none"
    # Identical in every setting except the redundant bit label.
    key = [c for c in (CONFIG_KEYS + ["seed", "__source__"])
           if c != "bits" and c in df.columns]
    return pd.concat([df[~unq], df[unq].drop_duplicates(subset=key)],
                     ignore_index=True)


def _ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (x.mean() - h, x.mean() + h)


def headline_table(df, metric):
    """Group by the full configuration; statistics over independent seeds.

    Every statistic is computed over one value per (configuration, seed).
    Historical CSVs re-run the same configuration in more than one file, and
    treating those rows as independent repetitions narrows the interval
    silently (the old code reported count=3 while averaging 6 rows). If the
    duplicates disagree, their mean is used and the disagreement is counted in
    the conflicts column, so it stays visible.
    """
    keys = [k for k in CONFIG_KEYS if k in df.columns]
    rows = []
    for key_vals, sub in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = dict(zip(keys, key_vals))
        if "seed" in sub.columns:
            per_seed = sub.groupby("seed", dropna=False)[metric]
            values = np.asarray(per_seed.mean().values, float)
            conflicts = int((per_seed.nunique() > 1).sum())
            dropped = int(len(sub) - len(values))
        else:
            values = np.asarray(sub[metric].values, float)
            conflicts = 0
            dropped = 0
        n = len(values)
        lo, hi = _ci95(values)
        rec.update(
            mean=float(values.mean()) if n else float("nan"),
            std=float(values.std(ddof=1)) if n > 1 else float("nan"),
            min=float(values.min()) if n else float("nan"),
            max=float(values.max()) if n else float("nan"),
            count=n, conflicts=conflicts, duplicate_rows=dropped,
            ci_lo=lo, ci_hi=hi,
        )
        rows.append(rec)
    g = pd.DataFrame(rows)
    g["mean_std"] = g.apply(
        lambda r: (f"{r['mean']:.4f} ± {r['std']:.4f}" if r["count"] > 1
                   else f"{r['mean']:.4f}"), axis=1)
    g["ci95"] = g.apply(
        lambda r: (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                   if r["count"] > 1 else ""), axis=1)
    return g


def _metric_for(task):
    if str(task).startswith("toy_regression"):
        return "best_test_loss"
    return "best_test_acc"


def _fmt_table(df, metric):
    cols = [c for c in ["task", "model", "optimizer", "grid_spec", "bits",
                        "epochs", "lr", "beta", "beta1", "k", "k_mode",
                        "step_scale", "project_lr_scale", "grad_noise_rho",
                        "accept_on", "mean_std", "ci95", "min", "max",
                        "count", "conflicts"] if c in df.columns]
    out = df[cols].to_markdown(index=False, floatfmt=".4f")
    return out.replace("mean_std", f"mean_std ({metric})")


# A "configuration family" fixes every setting except the bit-width, so each
# plotted line corresponds to one real method configuration. Averaging over
# beta / lr / noise settings (as the original plots did) produces lines that
# represent no method at all.
LINE_KEYS = [k for k in CONFIG_KEYS if k not in ("task", "model", "bits", "seed")]


def _fmt_val(v):
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _config_families(df, x_field):
    """Families that fix every setting except x_field and span >=2 values of it."""
    keys = [k for k in LINE_KEYS if k in df.columns and k != x_field]
    families = []
    for key_vals, g in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        g = g.dropna(subset=[x_field]).sort_values(x_field)
        if g[x_field].nunique() < 2:
            continue
        families.append((dict(zip(keys, key_vals)), g))
    return keys, families


def _varying_fields(families, keys):
    return {f: len({_fmt_val(kv.get(f)) for kv, _ in families}) > 1 for f in keys}


def _config_label(kv, varying, x_field):
    base = f"{kv.get('optimizer', '?')}/{kv.get('grid_spec', '?')}"
    extras = [f"{f}={_fmt_val(kv.get(f))}"
              for f, is_var in varying.items()
              if is_var and f not in ("optimizer", "grid_spec", x_field)]
    return base + ((" " + " ".join(extras)) if extras else "")


def _plot_generic(df, task, out_path, x_field, metric, ylabel, title,
                  only_gdmc=False):
    """One line per configuration family, varying x_field.

    Averaging over beta / lr / noise settings (as the original plots did)
    produces a line that represents no actual method, so every plotted line
    here fixes all settings except x_field.
    """
    sub = df[df["task"] == task]
    if only_gdmc:
        sub = sub[sub["optimizer"].str.startswith("gdmc")]
    if sub.empty or x_field not in sub.columns:
        out_path.unlink(missing_ok=True)
        return
    keys, families = _config_families(sub, x_field)
    if not families:
        # Nothing to draw for this axis; remove any stale figure.
        out_path.unlink(missing_ok=True)
        return
    varying = _varying_fields(families, keys)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for kv, g in families:
        agg = g.groupby(x_field)[metric].mean()
        ax.plot(agg.index.values, agg.values, "o-", alpha=0.85,
                label=_config_label(kv, varying, x_field))
    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (one line per configuration)")
    if x_field == "bits":
        ax.set_xscale("log", base=2)
    if metric == "final_acceptance_rate":
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right" if metric != "final_acceptance_rate" else "best",
              fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(df, task, out_path):
    _plot_generic(df, task, out_path, "bits", "best_test_acc",
                  "Best test accuracy",
                  f"{task}: best test accuracy vs quantization bits")


def plot_acceptance(df, task, out_path):
    _plot_generic(df, task, out_path, "bits", "final_acceptance_rate",
                  "Final acceptance rate (smoothed)",
                  f"{task}: GDMC acceptance rate vs quantization bits",
                  only_gdmc=True)


# When bits does not vary but something else does (noise strength, learning
# rate, step scale), emit a second figure for that axis.
ALT_X_FIELDS = ["grad_noise_rho", "lr", "step_scale", "project_lr_scale"]


def plot_alt_axis(df, task, plots_dir):
    sub = df[df["task"] == task]
    for xf in ALT_X_FIELDS:
        if xf not in sub.columns or sub[xf].nunique(dropna=True) < 2:
            continue
        out = plots_dir / f"{task}_vs_{xf}.png"
        _plot_generic(df, task, out, xf, "best_test_acc", "Best test accuracy",
                      f"{task}: best test accuracy vs {xf}")
        return


def write_report(df, out_path):
    lines = []
    lines.append("# GDMC for Deep Learning Weight Optimization - Final Report\n")
    lines.append("This report is generated entirely from results/raw/*.csv by "
                 "analysis/aggregate.py. Every row carries an explicit "
                 "configuration key (beta1, k, k_mode, step_scale, noise, "
                 "acceptance) and a 95% CI over seeds.\n")
    lines.append(f"Total runs: **{len(df)}** across "
                 f"{df['task'].nunique()} tasks.\n")
    lines.append("## Grouped results (all configurations)\n")
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        metric = _metric_for(task)
        lines.append(f"### {task}  (metric: {metric})\n")
        lines.append(_fmt_table(headline_table(sub, metric), metric) + "\n")
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

    tab = headline_table(df, "best_test_acc")
    tab.to_csv(out_dir / "headline_table.csv", index=False)
    n_conf = int(tab["conflicts"].sum()) if "conflicts" in tab.columns else 0
    if n_conf:
        print(f"WARNING: {n_conf} (config, seed) records appear more than once "
              f"with different results; their mean is used and the conflict is "
              f"recorded in the conflicts column.")
    md = ["# Headline table - best test accuracy by full configuration\n",
          "mean ± std and 95% CI over seeds. count = number of seeds.\n",
          "Unquantized baselines are de-duplicated to one row per run.\n",
          _fmt_table(tab, "best_test_acc") + "\n"]
    (out_dir / "headline_table.md").write_text("\n".join(md))

    for task in sorted(df["task"].unique()):
        plot_loss_curves(df, task, plots_dir / f"{task}_loss.png")
        plot_acceptance(df, task, plots_dir / f"{task}_acceptance.png")
        plot_alt_axis(df, task, plots_dir)

    write_report(df, out_dir / "REPORT.md")
    print(f"Wrote headline table ({len(tab)} configs), plots, and REPORT.md")


if __name__ == "__main__":
    main()
