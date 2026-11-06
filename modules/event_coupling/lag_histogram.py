"""
Lag histogram tools for Task 2 coupling analysis.

For every pair of events within +/- max_lag_ms in the same bout, the signed
time difference.

    lag = second_wave peak - first_wave peak

    collect_lags(deltas, ripples, "delta", "ripple"):
        negative lag -> ripple first (R->D)
        positive lag -> delta first  (D->R)
"""

import numpy as np
import pandas as pd

BOUT_KEYS = ["rat", "date", "trial", "nrem_bout_index"]

DEFAULT_EDGES = np.arange(-1000, 1020, 20)


def collect_lags(first_df, second_df, first_wave, second_wave, max_lag_ms=1000):
    """Signed time differences between all event pairs within the same NREM bout.

    Returns a flat array of (second_peak - first_peak) in ms.
    Pairs are only formed within a bout, never across bouts.
    """
    fp, sp = f"{first_wave}_peak_index", f"{second_wave}_peak_index"
    return collect_lags_cols(first_df, second_df, fp, sp, max_lag_ms=max_lag_ms)


def collect_lags_cols(first_df, second_df, first_col, second_col, max_lag_ms=1000):
    """Same as collect_lags, with the index columns given directly
    (e.g. ripple_end_index -> spindle_start_index).

    Returns a flat array of (second_col - first_col) in ms, paired within bout.
    """
    fp, sp = first_col, second_col

    f_groups = dict(tuple(first_df.dropna(subset=BOUT_KEYS + [fp]).groupby(BOUT_KEYS)))
    s_groups = dict(tuple(second_df.dropna(subset=BOUT_KEYS + [sp]).groupby(BOUT_KEYS)))

    out = []
    for key, fg in f_groups.items():
        sg = s_groups.get(key)
        if sg is None:
            continue

        fpk = fg[fp].to_numpy()
        spk = np.sort(sg[sp].to_numpy())

        # for each first event, the slice of second events inside the window
        lo = np.searchsorted(spk, fpk - max_lag_ms, "left")
        hi = np.searchsorted(spk, fpk + max_lag_ms, "right")
        n = hi - lo
        if n.sum() == 0:
            continue

        # expand the slices into one flat index array (no python loop)
        cum = np.cumsum(n) - n
        pos = np.arange(n.sum()) - np.repeat(cum, n) + np.repeat(lo, n)
        out.append(spk[pos] - np.repeat(fpk, n))

    return np.concatenate(out) if out else np.array([])


def summarise(
    lags, edges=DEFAULT_EDGES, baseline_ms=800, windows=None, label="", verbose=True
):
    """Turn an array of lags into baseline, peak and window ratios.

    baseline_ms : lags beyond +/- this value give the chance baseline
    windows     : list of (name, lo_ms, hi_ms), default R->D and D->R

    Returns a dict; also prints a one-line summary when verbose.
    """
    if windows is None:
        windows = [("R->D", -250, -50), ("D->R", 50, 400)]

    counts, _ = np.histogram(lags, bins=edges)
    left = edges[:-1]

    tail = (left < -baseline_ms) | (left >= baseline_ms)
    baseline = counts[tail].mean()

    res = {
        "label": label,
        "n_pairs": int(len(lags)),
        "counts": counts,
        "edges": edges,
        "baseline": float(baseline),
        "peak_ms": int(left[counts.argmax()]),
        "peak_height": int(counts.max()),
        "peak_pct_above": float(100 * (counts.max() - baseline) / baseline),
        "windows": {},
    }

    # window counts from the raw lags (bin edges don't fall on -250 / -50);
    # expected = per-bin baseline x window width in bins
    bin_w = float(edges[1] - edges[0])
    for name, lo, hi in windows:
        obs = int(np.sum((lags >= lo) & (lags < hi)))
        exp = baseline * (hi - lo) / bin_w
        res["windows"][name] = {
            "observed": obs,
            "expected": float(exp),
            "ratio": float(obs / exp) if exp > 0 else float("nan"),
        }

    if verbose:
        wtxt = " | ".join(f"{k} {v['ratio']:.3f}x" for k, v in res["windows"].items())
        print(
            f"{label:>12}: pairs {res['n_pairs']:>9,} | peak {res['peak_ms']:>5} ms "
            f"| +{res['peak_pct_above']:>5.1f}% above baseline | {wtxt}"
        )

    return res


def noise_floor_pct(res):
    """Approximate Poisson noise floor, as a percentage of baseline.

    100 / sqrt(baseline).
    """
    return 100 / np.sqrt(res["baseline"])


def self_test(planted_lag_ms=200, verbose=True):
    """Validate collect_lags on synthetic data with a known planted lag.

    If the recovered peak isn't the planted lag, there's an indexing bug.
    """
    rng = np.random.default_rng(1)

    d_t = np.arange(1000, 60000, 500)
    r_t = np.concatenate([d_t + planted_lag_ms, rng.integers(1000, 60000, 200)])

    syn_d = pd.DataFrame(
        {
            "rat": 1,
            "date": "x",
            "trial": 1,
            "nrem_bout_index": 1,
            "delta_peak_index": d_t,
        }
    )
    syn_r = pd.DataFrame(
        {
            "rat": 1,
            "date": "x",
            "trial": 1,
            "nrem_bout_index": 1,
            "ripple_peak_index": r_t,
        }
    )

    lags = collect_lags(syn_d, syn_r, "delta", "ripple")
    counts, _ = np.histogram(lags, bins=DEFAULT_EDGES)
    recovered = int(DEFAULT_EDGES[:-1][counts.argmax()])

    ok = recovered == planted_lag_ms
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(
            f"self_test: planted {planted_lag_ms} ms, recovered {recovered} ms -> {status}"
        )
    return ok
