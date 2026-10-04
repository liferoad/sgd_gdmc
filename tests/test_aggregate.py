"""Tests for the aggregator (regression for the pooled-configuration bug)."""
import json

import numpy as np
import pandas as pd

from analysis.aggregate import (_dedup_unquantized, _expand_extra, _ci95,
                                headline_table)


def make_df(beta1s=(0.0, 0.9), k=1, task="t", opts=("gdmc",), bits=8):
    rows = []
    for opt in opts:
        for b in beta1s:
            for s in (0, 1, 2):
                rows.append(dict(task=task, model="m", optimizer=opt,
                                 grid_spec="uniform", bits=bits, seed=s,
                                 beta1=b, k=k, best_test_acc=0.5 + 0.1 * b + 0.01 * s))
    return _expand_extra(pd.DataFrame(rows))


def test_different_beta1_are_not_pooled():
    t = headline_table(make_df(), "best_test_acc")
    assert len(t) == 2


def test_different_k_are_not_pooled():
    a = make_df(k=1)
    b = make_df(k=4)
    t = headline_table(pd.concat([a, b], ignore_index=True), "best_test_acc")
    assert len(t) == 4


def test_ci_columns_present_and_ordered():
    t = headline_table(make_df(), "best_test_acc")
    assert {"ci_lo", "ci_hi"}.issubset(t.columns)
    assert (t["ci_lo"] <= t["mean"]).all()
    assert (t["mean"] <= t["ci_hi"]).all()


def test_ci95_matches_t_interval():
    x = [0.9, 0.95, 1.0]
    lo, hi = _ci95(x)
    mean = np.mean(x)
    assert lo < mean < hi


def test_noise_runs_are_not_deduplicated():
    rows = []
    for rho in (0.0, 1.0, 3.0):
        for s in (0, 1, 2):
            rows.append(dict(task="t", model="m", optimizer="adam",
                             grid_spec="none", bits=32, seed=s,
                             best_test_acc=0.9 - 0.01 * rho, epochs=8,
                             lr=1e-3, batch_size=128,
                             extra=json.dumps({"grad_noise_rho": rho})))
    df = _dedup_unquantized(_expand_extra(pd.DataFrame(rows)))
    assert len(df) == 9, "noisy runs were collapsed by de-duplication"
    t = headline_table(df, "best_test_acc")
    assert len(t) == 3
    assert set(t["grad_noise_rho"]) == {0.0, 1.0, 3.0}


def test_acceptance_ablation_is_not_pooled():
    rows = []
    for beta in (0.0, 2.0):
        for s in (0, 1, 2):
            rows.append(dict(task="t", model="m", optimizer="gdmc",
                             grid_spec="uniform", bits=8, seed=s, beta=beta,
                             beta1=0.9, k=1, best_test_acc=0.95 + 0.001 * s))
    t = headline_table(_expand_extra(pd.DataFrame(rows)), "best_test_acc")
    assert len(t) == 2, "beta=0 (always accept) and beta=2 were pooled"
    assert set(t["count"]) == {3}
    assert set(t["beta"]) == {0.0, 2.0}


def test_learning_rate_sweeps_are_not_pooled():
    rows = []
    for lr in (1e-3, 3e-3):
        for s in (0, 1, 2):
            rows.append(dict(task="t", model="m", optimizer="adam",
                             grid_spec="none", bits=32, seed=s, lr=lr,
                             best_test_acc=0.98, epochs=10, batch_size=128))
    t = headline_table(_expand_extra(pd.DataFrame(rows)), "best_test_acc")
    assert len(t) == 2


def test_unquantized_baselines_are_deduplicated():
    rows = []
    for bits in (2, 8, 32):
        rows.append(dict(task="t", model="m", optimizer="adam", grid_spec="none",
                         bits=bits, seed=0, best_test_acc=0.9, epochs=30,
                         lr=1e-3, batch_size=128))
    df = _expand_extra(pd.DataFrame(rows))
    out = _dedup_unquantized(df)
    assert len(out) == 1
