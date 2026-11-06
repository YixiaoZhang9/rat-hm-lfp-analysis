"""
surrogate.py
Circular-shift null for the coupling analyses.

One event train (the ripples, the smaller one) is moved forward by a random
amount and wrapped around, so its own timing is kept and only the alignment
with the other train is lost.

scope="bout"       shift within each NREM bout (short bouts are skipped)
scope="recording"  shift within the recording's NREM time (Yixiao, 30 July 2026:
                   per wave, per area, per recording date)

Minimum shift 1 s (the observed peak is ~140 ms).
"""

import numpy as np
import pandas as pd

BOUT_KEYS = ["rat", "date", "trial", "nrem_bout_index"]


def circular_shift_peaks(peaks, span_lo, span_hi, rng, min_shift_ms=1000):
    """Shift peak times circularly within [span_lo, span_hi).

    Returns shifted peaks, or None if the span is too short to allow a shift
    that clears min_shift_ms at both ends.
    """
    span = span_hi - span_lo
    if span <= 2 * min_shift_ms:
        return None

    shift = rng.uniform(min_shift_ms, span - min_shift_ms)
    return span_lo + np.mod(peaks - span_lo + shift, span)


def _bout_spans(df, peak_col):
    """Per-bout (start, end) to wrap within.

    Uses nrem_bout_start_index / _end_index if present, otherwise the range of
    the observed peaks.
    """
    have_bounds = (
        "nrem_bout_start_index" in df.columns
        and "nrem_bout_end_index" in df.columns
        and df["nrem_bout_start_index"].notna().any()
    )
    if have_bounds:
        g = df.groupby(BOUT_KEYS).agg(
            lo=("nrem_bout_start_index", "first"), hi=("nrem_bout_end_index", "first")
        )
    else:
        g = df.groupby(BOUT_KEYS).agg(lo=(peak_col, "min"), hi=(peak_col, "max"))
    return g


def _shift_within_recording(sg, sp, rng, min_shift_ms):
    """Circular shift across a whole recording, staying inside NREM.

    The bouts are put end to end, the shift is done on that timeline and the
    times are mapped back, so events stay in NREM.

    Returns a DataFrame with shifted peaks and their new bout, or None if the
    recording is too short.
    """
    bouts = (
        sg[["nrem_bout_index", "nrem_bout_start_index", "nrem_bout_end_index"]]
        .drop_duplicates()
        .sort_values("nrem_bout_start_index")
    )
    if bouts.empty:
        return None

    lo = bouts["nrem_bout_start_index"].to_numpy(dtype=float)
    hi = bouts["nrem_bout_end_index"].to_numpy(dtype=float)
    idx = bouts["nrem_bout_index"].to_numpy()

    lengths = hi - lo
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])  # offset of each bout
    total = float(lengths.sum())
    if total <= 2 * min_shift_ms:
        return None

    # event time -> offset on the concatenated NREM timeline
    pos = np.searchsorted(idx, sg["nrem_bout_index"].to_numpy())
    pos = np.clip(pos, 0, len(idx) - 1)
    offset = starts[pos] + (sg[sp].to_numpy(dtype=float) - lo[pos])

    shift = rng.uniform(min_shift_ms, total - min_shift_ms)
    new_off = np.mod(offset + shift, total)

    # offset -> back to a real time inside whichever bout it lands in
    j = np.searchsorted(starts, new_off, side="right") - 1
    j = np.clip(j, 0, len(starts) - 1)

    out = sg.copy()
    out[sp] = lo[j] + (new_off - starts[j])
    out["nrem_bout_index"] = idx[j]
    return out


def shifted_lags(
    first_df,
    second_df,
    first_wave,
    second_wave,
    rng,
    edges,
    max_lag_ms=1000,
    min_shift_ms=1000,
    scope="bout",
):
    """One surrogate iteration: circular shift of the second train.

    Returns the histogram counts and the number of units used.
    """
    fp, sp = f"{first_wave}_peak_index", f"{second_wave}_peak_index"
    return shifted_lags_cols(
        first_df,
        second_df,
        fp,
        sp,
        rng,
        edges,
        max_lag_ms=max_lag_ms,
        min_shift_ms=min_shift_ms,
        scope=scope,
    )


def shifted_lags_cols(
    first_df,
    second_df,
    first_col,
    second_col,
    rng,
    edges,
    max_lag_ms=1000,
    min_shift_ms=1000,
    scope="bout",
):
    """Same as shifted_lags, with the index columns given directly
    (e.g. ripple_end_index -> spindle_start_index)."""
    fp, sp = first_col, second_col

    f = first_df.dropna(subset=BOUT_KEYS + [fp])
    s = second_df.dropna(subset=BOUT_KEYS + [sp])

    if scope == "recording":
        return _shifted_lags_recording(
            f, s, fp, sp, rng, edges, max_lag_ms, min_shift_ms
        )

    spans = _bout_spans(s, sp)
    s_groups = dict(tuple(s.groupby(BOUT_KEYS)))

    out = []
    used = 0
    for key, fg in f.groupby(BOUT_KEYS):
        sg = s_groups.get(key)
        if sg is None:
            continue
        if key not in spans.index:
            continue

        lo, hi = spans.loc[key, "lo"], spans.loc[key, "hi"]
        spk = circular_shift_peaks(sg[sp].to_numpy(), lo, hi, rng, min_shift_ms)
        if spk is None:
            continue  # bout too short to shift
        used += 1

        spk = np.sort(spk)
        fpk = fg[fp].to_numpy()

        a = np.searchsorted(spk, fpk - max_lag_ms, "left")
        b = np.searchsorted(spk, fpk + max_lag_ms, "right")
        n = b - a
        if n.sum() == 0:
            continue
        cum = np.cumsum(n) - n
        pos = np.arange(n.sum()) - np.repeat(cum, n) + np.repeat(a, n)
        out.append(spk[pos] - np.repeat(fpk, n))

    lags = np.concatenate(out) if out else np.array([])
    counts, _ = np.histogram(lags, bins=edges)
    return counts, used


REC_KEYS = ["rat", "date", "trial"]


def _shifted_lags_recording(f, s, fp, sp, rng, edges, max_lag_ms, min_shift_ms):
    """Recording-scope version: shift on the concatenated NREM timeline."""
    f_recs = dict(tuple(f.groupby(REC_KEYS)))
    out = []
    used = 0

    for key, sg in s.groupby(REC_KEYS):
        fg_all = f_recs.get(key)
        if fg_all is None:
            continue

        shifted = _shift_within_recording(sg, sp, rng, min_shift_ms)
        if shifted is None:
            continue
        used += 1

        # pair only inside the same bout, same as collect_lags
        s_by_bout = dict(tuple(shifted.groupby("nrem_bout_index")))
        for b, fg in fg_all.groupby("nrem_bout_index"):
            sb = s_by_bout.get(b)
            if sb is None:
                continue
            spk = np.sort(sb[sp].to_numpy())
            fpk = fg[fp].to_numpy()

            a = np.searchsorted(spk, fpk - max_lag_ms, "left")
            bb = np.searchsorted(spk, fpk + max_lag_ms, "right")
            n = bb - a
            if n.sum() == 0:
                continue
            cum = np.cumsum(n) - n
            pos = np.arange(n.sum()) - np.repeat(cum, n) + np.repeat(a, n)
            out.append(spk[pos] - np.repeat(fpk, n))

    lags = np.concatenate(out) if out else np.array([])
    counts, _ = np.histogram(lags, bins=edges)
    return counts, used


def surrogate_distribution(
    first_df,
    second_df,
    first_wave,
    second_wave,
    edges,
    n_iter=1000,
    seed=0,
    min_shift_ms=1000,
    max_lag_ms=1000,
    scope="bout",
    verbose=True,
    checkpoint_every=None,
    checkpoint_fn=None,
    resume_from=None,
):
    """Build the null distribution of histogram counts.

    checkpoint_every / checkpoint_fn: checkpoint_fn(partial) is called every
        `checkpoint_every` iterations with the results so far.
    resume_from: earlier result or checkpoint; its iterations are reused and
        the RNG is advanced, so the result equals a fresh run.

    Returns dict with:
        null       : (n_iter, n_bins) array of surrogate counts
        real       : observed counts on the same bins
        z          : per-bin z-score of observed vs. the null
        n_bouts    : bouts used per iteration (same for every iteration)
        n_done     : number of iterations with data (== n_iter unless it's
                     an interrupted checkpoint)
    """
    fp, sp = f"{first_wave}_peak_index", f"{second_wave}_peak_index"
    return surrogate_distribution_cols(
        first_df,
        second_df,
        fp,
        sp,
        edges,
        n_iter=n_iter,
        seed=seed,
        min_shift_ms=min_shift_ms,
        max_lag_ms=max_lag_ms,
        scope=scope,
        verbose=verbose,
        checkpoint_every=checkpoint_every,
        checkpoint_fn=checkpoint_fn,
        resume_from=resume_from,
    )


def surrogate_distribution_cols(
    first_df,
    second_df,
    first_col,
    second_col,
    edges,
    n_iter=1000,
    seed=0,
    min_shift_ms=1000,
    max_lag_ms=1000,
    scope="bout",
    verbose=True,
    checkpoint_every=None,
    checkpoint_fn=None,
    resume_from=None,
):
    """Same as surrogate_distribution, with the index columns given directly
    (e.g. ripple_end_index -> spindle_start_index)."""
    from .lag_histogram import collect_lags_cols

    real_lags = collect_lags_cols(
        first_df, second_df, first_col, second_col, max_lag_ms=max_lag_ms
    )
    real, _ = np.histogram(real_lags, bins=edges)

    null = np.zeros((n_iter, len(edges) - 1))
    start = 0
    rng = np.random.default_rng(seed)

    if resume_from is not None:
        prev_n = int(resume_from.get("n_done", 0))
        prev_n = min(prev_n, n_iter)
        null[:prev_n] = resume_from["null"][:prev_n]
        start = prev_n
        if verbose:
            print(f"  resuming from checkpoint: {start}/{n_iter} already done")
        # advance the RNG past the iterations already done
        for _ in range(start):
            shifted_lags_cols(
                first_df,
                second_df,
                first_col,
                second_col,
                rng,
                edges,
                max_lag_ms=max_lag_ms,
                min_shift_ms=min_shift_ms,
                scope=scope,
            )

    used = 0
    for i in range(start, n_iter):
        null[i], used = shifted_lags_cols(
            first_df,
            second_df,
            first_col,
            second_col,
            rng,
            edges,
            max_lag_ms=max_lag_ms,
            min_shift_ms=min_shift_ms,
            scope=scope,
        )
        done = i + 1
        if verbose and done % 100 == 0:
            print(f"  {done}/{n_iter}")
        if checkpoint_every and checkpoint_fn and done % checkpoint_every == 0:
            partial_mu = null[:done].mean(axis=0)
            partial_sd = null[:done].std(axis=0)
            partial_sd_safe = partial_sd.copy()
            partial_sd_safe[partial_sd_safe == 0] = np.nan
            checkpoint_fn(
                {
                    "null": null.copy(),
                    "real": real,
                    "z": (real - partial_mu) / partial_sd_safe,
                    "mu": partial_mu,
                    "sd": partial_sd,
                    "n_bouts": used,
                    "edges": edges,
                    "n_done": done,
                }
            )

    mu = null.mean(axis=0)
    sd = null.std(axis=0)
    sd[sd == 0] = np.nan
    z = (real - mu) / sd

    return {
        "null": null,
        "real": real,
        "z": z,
        "mu": mu,
        "sd": sd,
        "n_bouts": used,
        "edges": edges,
        "n_done": n_iter,
    }


def window_significance(dist, edges, window_ms, one_sided=True):
    """One-sided permutation test on the total count inside a window.

    Yixiao (2 Sep 2026): sum over the whole window instead of per-bin z-scores.

    dist       : output of surrogate_distribution (has 'null' and 'real')
    edges      : same bin edges used to build dist
    window_ms  : (lo, hi), e.g. (-250, -50) for R->D

    p = (1 + hits) / (1 + n_iter), so with 1000 iterations the minimum is 1/1001.
    """
    left = edges[:-1]
    lo, hi = window_ms
    mask = (left >= lo) & (left < hi)
    if not mask.any():
        raise ValueError(f"window {window_ms} does not overlap the bin edges")

    obs = int(dist["real"][mask].sum())
    null_sums = dist["null"][:, mask].sum(axis=1)

    if one_sided:
        p = (1 + np.sum(null_sums >= obs)) / (1 + len(null_sums))
    else:
        p = (
            1
            + np.sum(
                np.abs(null_sums - null_sums.mean()) >= abs(obs - null_sums.mean())
            )
        ) / (1 + len(null_sums))

    return {
        "observed": obs,
        "null_sums": null_sums,
        "null_mean": float(null_sums.mean()),
        "null_sd": float(null_sums.std()),
        "p": float(p),
        "significant": p < 0.05,
        "window_ms": window_ms,
    }


def surrogate_unique_count(
    first_df,
    second_df,
    first_wave,
    second_wave,
    window_ms=None,
    n_iter=1000,
    seed=0,
    min_shift_ms=1000,
    scope="recording",
    verbose=True,
):
    """Null distribution for the unique-ripple count (Yixiao's rule).

    Used instead of the analytic 1 - exp(-r*w), which assumes randomly timed
    delta waves.

    Returns dict with:
        real     : observed unique-ripple count
        null     : array of n_iter surrogate counts
        z        : (real - null mean) / null sd
        p        : one-sided empirical p, (1 + #{null >= real}) / (1 + n_iter)
        ratio    : real / null mean
        n_ripple : total ripples used
    """
    from .coupling_unique import count_sequences_unique

    obs = count_sequences_unique(
        first_df, second_df, first_wave, second_wave, window_ms=window_ms
    )
    real = int(obs["n_sequences"].sum())
    n_rip = int(
        obs[f"n_{second_wave}"].sum()
        if second_wave == "ripple"
        else obs[f"n_{first_wave}"].sum()
    )

    sp = f"{second_wave}_peak_index"
    s = second_df.dropna(subset=BOUT_KEYS + [sp])

    rng = np.random.default_rng(seed)
    null = np.zeros(n_iter)

    for i in range(n_iter):
        parts = []
        if scope == "recording":
            for _, sg in s.groupby(REC_KEYS):
                sh = _shift_within_recording(sg, sp, rng, min_shift_ms)
                if sh is not None:
                    parts.append(sh)
        else:
            spans = _bout_spans(s, sp)
            for key, sg in s.groupby(BOUT_KEYS):
                if key not in spans.index:
                    continue
                lo, hi = spans.loc[key, "lo"], spans.loc[key, "hi"]
                pk = circular_shift_peaks(sg[sp].to_numpy(), lo, hi, rng, min_shift_ms)
                if pk is None:
                    continue
                sh = sg.copy()
                sh[sp] = pk
                parts.append(sh)

        if not parts:
            null[i] = np.nan
            continue

        shifted = pd.concat(parts, ignore_index=True)
        null[i] = count_sequences_unique(
            first_df, shifted, first_wave, second_wave, window_ms=window_ms
        )["n_sequences"].sum()

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_iter}")

    mu, sd = np.nanmean(null), np.nanstd(null)
    p = (1 + np.nansum(null >= real)) / (1 + np.sum(~np.isnan(null)))

    return {
        "real": real,
        "null": null,
        "mu": float(mu),
        "sd": float(sd),
        "z": float((real - mu) / sd) if sd > 0 else np.nan,
        "p": float(p),
        "ratio": float(real / mu) if mu > 0 else np.nan,
        "n_ripple": n_rip,
    }


# self tests, run with: python surrogate.py


def _synth(
    n_bouts=60, bout_ms=80_000, n_delta=240, planted_lag=None, coupled_frac=0.35, seed=0
):
    """Synthetic deltas (~1 Hz) and ripples, coupled or not."""
    rng = np.random.default_rng(seed)
    D, R = [], []
    for b in range(n_bouts):
        # deltas around 1 Hz with some jitter
        d = np.cumsum(rng.normal(bout_ms / n_delta, bout_ms / n_delta * 0.3, n_delta))
        d = d[d < bout_ms]

        # clustered ripples, independent of the deltas
        n_clust = 12
        centres = rng.uniform(0, bout_ms, n_clust)
        r = np.concatenate([c + rng.normal(0, 400, 6) for c in centres])
        r = r[(r > 0) & (r < bout_ms)]

        if planted_lag is not None:
            take = rng.choice(len(d), int(coupled_frac * len(d)), replace=False)
            r = np.concatenate(
                [r, d[take] + planted_lag + rng.normal(0, 25, len(take))]
            )
            r = r[(r > 0) & (r < bout_ms)]

        for arr, sink in ((d, D), (np.sort(r), R)):
            sink.append(
                pd.DataFrame(
                    {
                        "rat": 1,
                        "date": "x",
                        "trial": 1,
                        "nrem_bout_index": b,
                        "nrem_bout_start_index": 0.0,
                        "nrem_bout_end_index": float(bout_ms),
                        "peak": arr,
                    }
                )
            )

    dd = pd.concat(D, ignore_index=True).rename(columns={"peak": "delta_peak_index"})
    rr = pd.concat(R, ignore_index=True).rename(columns={"peak": "ripple_peak_index"})
    return dd, rr


if __name__ == "__main__":
    edges = np.arange(-1000, 1020, 20)
    left = edges[:-1]
    win = (left >= -250) & (left < -50)

    print("TEST 1 -- no real coupling: the null must NOT flag anything")
    d, r = _synth(planted_lag=None, seed=1)
    res = surrogate_distribution(
        d, r, "delta", "ripple", edges, n_iter=200, seed=1, verbose=False
    )
    print(f"  bouts used      : {res['n_bouts']}")
    print(f"  max |z| any bin : {np.nanmax(np.abs(res['z'])):.2f}   (want < ~4)")
    print(f"  mean z in window: {np.nanmean(res['z'][win]):.2f}   (want near 0)")

    print("\nTEST 2 -- ripples planted 140 ms BEFORE deltas: must be flagged")
    d, r = _synth(planted_lag=-140, seed=2)
    res = surrogate_distribution(
        d, r, "delta", "ripple", edges, n_iter=200, seed=2, verbose=False
    )
    peak_bin = int(left[np.nanargmax(res["z"])])
    print(f"  peak z at       : {peak_bin} ms   (want -140)")
    print(f"  z at that bin   : {np.nanmax(res['z']):.1f}   (want large)")
    print(f"  mean z in window: {np.nanmean(res['z'][win]):.1f}")

    print("\nTEST 3 -- does the shift preserve the train's own structure?")
    rng = np.random.default_rng(0)
    S = 60_000
    pk = np.sort(rng.uniform(0, S, 300))
    sh = np.sort(circular_shift_peaks(pk, 0, S, rng))

    def cyclic_gaps(p, lo, hi):
        """Gaps on the circle, including the gap across the wrap point."""
        return np.sort(np.concatenate([np.diff(p), [(hi - p[-1]) + (p[0] - lo)]]))

    same = np.allclose(cyclic_gaps(pk, 0, S), cyclic_gaps(sh, 0, S), atol=1e-6)
    print(f"  cyclic inter-event intervals identical: {same}   (want True)")

    print("\nTEST 4 -- unique-ripple surrogate, no real coupling")
    d, r = _synth(planted_lag=None, seed=3)
    u = surrogate_unique_count(
        d, r, "delta", "ripple", n_iter=100, seed=3, verbose=False
    )
    print(
        f"  observed {u['real']} of {u['n_ripple']} ripples | "
        f"null mean {u['mu']:.0f} | ratio {u['ratio']:.3f} | z {u['z']:+.2f} "
        f"| p {u['p']:.3f}   (want ratio ~1, p not small)"
    )

    print("\nTEST 5 -- unique-ripple surrogate, 140 ms lag planted")
    d, r = _synth(planted_lag=-140, seed=4)
    u = surrogate_unique_count(
        d, r, "delta", "ripple", n_iter=100, seed=4, verbose=False
    )
    print(
        f"  observed {u['real']} of {u['n_ripple']} ripples | "
        f"null mean {u['mu']:.0f} | ratio {u['ratio']:.3f} | z {u['z']:+.2f} "
        f"| p {u['p']:.3f}   (want ratio > 1, p small)"
    )

    print("\nTEST 6 -- is the analytic chance level right?")
    from .coupling_unique import expected_by_chance

    d, r = _synth(planted_lag=None, seed=5)
    total_s = 60 * 80_000 / 1000.0  # 60 bouts x 80 s
    exp_a, p_a = expected_by_chance(len(r), len(d) / total_s, (50, 400))
    u = surrogate_unique_count(
        d, r, "delta", "ripple", n_iter=100, seed=5, verbose=False
    )
    print(f"  analytic 1-exp(-r*w) : {exp_a:>7.0f}  (P = {100 * p_a:.1f}%)")
    print(f"  circular-shift null  : {u['mu']:>7.0f}")
    print(
        f"  analytic / surrogate : {exp_a / u['mu']:.3f}   "
        "(1.0 would mean the formula is safe to use)"
    )
