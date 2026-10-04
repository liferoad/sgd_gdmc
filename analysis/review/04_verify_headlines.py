"""Review probe 4: recompute the claims used in docs/review_2026-10-04.md.

Reads only committed artifacts (results/raw/*.csv, results/curves/*).

Usage: .venv/bin/python analysis/review/04_verify_headlines.py
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (x.mean() - h, x.mean() + h)


def main():
    raw = ROOT / "results" / "raw"
    all_rows = sum(len(pd.read_csv(f)) for f in sorted(raw.glob("*.csv")))
    print(f"[1] raw rows across results/raw/*.csv: {all_rows} "
          f"(docs/test_report.md claims 327)")

    # --- acceptance fractions from the committed per-step curves ---
    fr = []
    for f in sorted((ROOT / "results" / "curves" / "mnist_mlp_v3").glob("*gdmc*.csv")):
        d = pd.read_csv(f)
        a = d["accepted"].dropna().astype(float)
        if len(a):
            fr.append(a.mean())
    print(f"[2] MNIST MLP v3 GDMC curves: n={len(fr)}  "
          f"median accepted={np.median(fr):.4f}  mean={np.mean(fr):.4f}  "
          f"min={np.min(fr):.4f}  files<0.99: {sum(np.array(fr)<0.99)}")

    # --- Adam learning rate in the long-run baseline vs the v3 script ---
    lr = pd.read_csv(raw / "long_runs_mlp.csv")
    adam_long = lr[lr.optimizer == "adam"]["best_test_acc"].values
    mom_long = lr[lr.optimizer == "momentum"]["best_test_acc"].values
    v3 = pd.read_csv(raw / "gdmc_v3_auto_k.csv")
    adam_v3 = v3[v3.optimizer == "adam"]["best_test_acc"].values
    print(f"[3] Adam best test acc: long_runs_mlp lr=1e-2 = "
          f"{adam_long.mean():.4f} ± {adam_long.std(ddof=1):.4f}; "
          f"v3 script lr=1e-3 = {adam_v3.mean():.4f} ± {adam_v3.std(ddof=1):.4f}; "
          f"momentum lr=1e-2 = {mom_long.mean():.4f}")

    print("[4] MNIST MLP v3 script, GDMC vs correctly-tuned Adam "
          "(seed-matched, Welch t-test):")
    v3 = v3.copy()
    v3["ss"] = v3["extra"].apply(lambda s: json.loads(s).get("step_scale"))
    for (opt, bits, ss), g in v3.groupby(["optimizer", "bits", "ss"]):
        if opt == "adam":
            continue
        b = g["best_test_acc"].values
        if len(b) < 2:
            continue
        p = stats.ttest_ind(b, adam_v3, equal_var=False).pvalue
        print(f"    {opt:12s} b{bits:<2d} ss={ss:<7g} mean={b.mean():.4f} "
              f"delta={b.mean()-adam_v3.mean():+.4f} p={p:.3f}")

    # --- Projected-GD lr dependence (probe 3 output) ---
    pc = ROOT / "results" / "review" / "projected_lr_check.csv"
    if pc.exists():
        d = pd.read_csv(pc)
        print("[5] Projected-GD lr sweep (MNIST MLP 10K, 5 epochs, 3 seeds):")
        print(d.groupby(["optimizer", "bits", "lr"])["best_test_acc"]
               .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4f}"))
    else:
        print("[5] results/review/projected_lr_check.csv missing; "
              "run 03_projected_lr_check.py")


if __name__ == "__main__":
    main()
