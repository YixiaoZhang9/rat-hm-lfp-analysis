"""
overtime_stats.py
Trend statistics for the over-time analyses.

Slide 11 of the analysis plan:
  * per rat: Spearman between the time axis and the measure
  * per rat: permutation p from shuffling day order
  * across rats: Wilcoxon signed-rank on the rho values against 0
  * Benjamini-Hochberg FDR

With 7 rats Wilcoxon can't go below p ~0.016 and needs 6 non-zero values.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def permute_trend(x, y, n_iter=1000, seed=0):
    """Permutation p-value for a Spearman trend, by shuffling the time axis.

    Two-tailed. Returns (rho, p_perm, null_distribution).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return np.nan, np.nan, np.array([])

    rho, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)

    null = np.empty(n_iter)
    for i in range(n_iter):
        null[i] = spearmanr(x, rng.permutation(y))[0]

    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + n_iter)
    return float(rho), float(p), null


def trend_per_rat(
    table, x_col, y_col="ratio", group_cols=("area",), min_points=5, n_iter=1000, seed=0
):
    """Spearman trend per rat, with a permutation p-value for each.

    table       one row per rat per recording day (per area)
    x_col       the time axis: days_elapsed, gl_num, session_num ...
    y_col       the measure being tracked
    group_cols  computed separately within each combination of these

    Returns one row per rat per group.
    """
    rows = []
    keys = list(group_cols) + ["rat"]
    for key, g in table.groupby(keys, observed=True):
        g = g.dropna(subset=[x_col, y_col])
        if len(g) < min_points:
            continue
        rho, p_perm, _ = permute_trend(g[x_col], g[y_col], n_iter=n_iter, seed=seed)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update(
            {
                "n_days": len(g),
                "rho": rho,
                "p_perm": p_perm,
                "first": g.sort_values(x_col)[y_col].iloc[:3].mean(),
                "last": g.sort_values(x_col)[y_col].iloc[-3:].mean(),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def across_rats(trend_table, group_cols=("area",), verbose=True):
    """Wilcoxon signed-rank on the per-rat correlations against zero."""
    out = []
    # a single key as a string avoids a pandas FutureWarning
    gc = list(group_cols)
    by = gc[0] if len(gc) == 1 else gc
    for key, g in trend_table.groupby(by, observed=True):
        rec = dict(zip(gc, key if isinstance(key, tuple) else (key,)))
        n = len(g)
        rec["n_rats"] = n
        rec["median_rho"] = g["rho"].median()
        rec["n_positive"] = int((g["rho"] > 0).sum())

        # keep scipy's small-n / ties warning with the result
        note = ""
        if n >= 3 and g["rho"].abs().sum() > 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    _, p = wilcoxon(g["rho"])
                except ValueError:
                    p = 1.0
            msgs = [str(w.message) for w in caught]
            if any("ties" in m for m in msgs):
                note = "ties"
            if any("too small" in m for m in msgs):
                note = (note + "+approx") if note else "approx"
        else:
            p = np.nan
        rec["p_wilcoxon"] = p
        rec["p_note"] = note  # blank = exact test, otherwise unreliable
        out.append(rec)

    res = pd.DataFrame(out)
    if verbose and len(res):
        print(res.round(4).to_string(index=False))
        if res["n_rats"].max() < 8:
            print(
                f"\n  note: n = {res['n_rats'].max()} rats. Wilcoxon cannot return "
                "p below ~0.016 at this sample size,"
            )
            print(
                "  and needs >= 6 non-zero values to run. A null result here is not "
                "evidence of absence."
            )
    return res


def fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full(len(p), np.nan)
    if ok.sum() == 0:
        return adj

    sub = p[ok]
    n = len(sub)
    order = np.argsort(sub)
    a = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        val = sub[i] * n / (n - rank + 1)
        prev = min(prev, val)
        a[i] = prev
    adj[ok] = np.minimum(a, 1.0)
    return adj


# self tests: python overtime_stats.py

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("TEST 1 -- no trend: permutation p should NOT be small")
    x = np.arange(20)
    y = rng.normal(1.2, 0.06, 20)
    rho, p, _ = permute_trend(x, y, n_iter=500, seed=1)
    print(f"  rho {rho:+.3f} | p_perm {p:.3f}   (want p not small)")

    print("\nTEST 2 -- planted rising trend: must be detected")
    y = 1.1 + 0.008 * x + rng.normal(0, 0.03, 20)
    rho, p, _ = permute_trend(x, y, n_iter=500, seed=2)
    print(f"  rho {rho:+.3f} | p_perm {p:.3f}   (want rho high, p small)")

    print("\nTEST 3 -- across rats, 7 rats all trending up")
    rows = []
    for rat in range(1, 8):
        xs = np.arange(18)
        ys = 1.1 + 0.006 * xs + rng.normal(0, 0.04, 18)
        rows += [
            {"rat": rat, "area": "PL", "days_elapsed": a, "ratio": b}
            for a, b in zip(xs, ys)
        ]
    t = trend_per_rat(pd.DataFrame(rows), "days_elapsed", n_iter=200)
    print(t.round(3).to_string(index=False))
    across_rats(t)

    print("\nTEST 4 -- across rats, no real trend")
    rows = []
    for rat in range(1, 8):
        xs = np.arange(18)
        ys = rng.normal(1.15, 0.05, 18)
        rows += [
            {"rat": rat, "area": "PL", "days_elapsed": a, "ratio": b}
            for a, b in zip(xs, ys)
        ]
    t = trend_per_rat(pd.DataFrame(rows), "days_elapsed", n_iter=200)
    print()
    across_rats(t)

    print("\nTEST 5 -- FDR against a hand-computed example")
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    want = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205]
    got = fdr(p)
    print(f"  matches: {np.allclose(got, want, atol=1e-3)}")
