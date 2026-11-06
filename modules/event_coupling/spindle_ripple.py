"""
spindle_ripple.py
Ripples inside spindles ("spindle with ripple").

rule="peak_in_spindle"   ripple peak between spindle start and end (default)
rule="any_overlap"       ripple interval overlaps the spindle at all

Ripples are HPC only: HPC ripple in PL / RSC spindle (cross-channel) and in
HPC spindle (same channel, check only).

The analytic chance level (share of NREM time covered by spindles) is only
descriptive, use `surrogate_overlap` for the stats.
"""

import numpy as np
import pandas as pd

BOUT_KEYS = ["rat", "date", "trial", "nrem_bout_index"]


def count_overlap(spindles, ripples, rule="peak_in_spindle"):
    """Which ripples fall inside a spindle? One row per bout.

    Returns: BOUT_KEYS + n_spindle, n_ripple, n_coupled, spindle_ms
      n_coupled          : ripples inside at least one spindle (each counted once)
      n_spindle_coupled  : spindles with at least one ripple (Navarro Lobato)
      spindle_ms         : total spindle time in the bout
    """
    sp = spindles.dropna(
        subset=BOUT_KEYS + ["spindle_start_index", "spindle_end_index"]
    )
    rp = ripples.dropna(subset=BOUT_KEYS + ["ripple_peak_index"])

    need_rip_interval = rule == "any_overlap"
    if need_rip_interval:
        rp = rp.dropna(subset=["ripple_start_index", "ripple_end_index"])

    sp_groups = dict(tuple(sp.groupby(BOUT_KEYS)))

    rows = []
    for key, rg in rp.groupby(BOUT_KEYS):
        sg = sp_groups.get(key)
        n_rip = len(rg)
        if sg is None or sg.empty:
            rows.append((*key, 0, n_rip, 0, 0, 0.0))
            continue

        s_lo = sg["spindle_start_index"].to_numpy(dtype=float)
        s_hi = sg["spindle_end_index"].to_numpy(dtype=float)
        order = np.argsort(s_lo)
        s_lo, s_hi = s_lo[order], s_hi[order]

        # running max of end times (spindles can overlap)
        s_hi_max = np.maximum.accumulate(s_hi)

        if rule == "peak_in_spindle":
            r_lo = r_hi = rg["ripple_peak_index"].to_numpy(dtype=float)
        elif rule == "any_overlap":
            r_lo = rg["ripple_start_index"].to_numpy(dtype=float)
            r_hi = rg["ripple_end_index"].to_numpy(dtype=float)
        else:
            raise ValueError(f"unknown rule: {rule}")

        # coupled if a spindle overlaps [ripple start, ripple end]
        idx = np.searchsorted(s_lo, r_hi, side="right")  # spindles starting in time
        coupled = np.zeros(len(r_lo), dtype=bool)
        has_any = idx > 0
        coupled[has_any] = s_hi_max[idx[has_any] - 1] >= r_lo[has_any]

        # exact check for the few where the running max says "maybe"
        for i in np.flatnonzero(coupled):
            j = idx[i]
            coupled[i] = np.any((s_lo[:j] <= r_hi[i]) & (s_hi[:j] >= r_lo[i]))

        # spindle level: spindles with at least one ripple (Navarro Lobato)
        sp_hit = np.zeros(len(s_lo), dtype=bool)
        for i in range(len(r_lo)):
            sp_hit |= (s_lo <= r_hi[i]) & (s_hi >= r_lo[i])

        spindle_ms = float((s_hi - s_lo).sum())
        rows.append(
            (*key, len(sg), n_rip, int(coupled.sum()), int(sp_hit.sum()), spindle_ms)
        )

    return pd.DataFrame(
        rows,
        columns=BOUT_KEYS
        + ["n_spindle", "n_ripple", "n_coupled", "n_spindle_coupled", "spindle_ms"],
    )


def analytic_chance(overlap_table, total_nrem_s):
    """Fraction of ripples expected inside a spindle if timing were random.

    Share of NREM time covered by spindles (descriptive only).
    """
    spindle_s = overlap_table["spindle_ms"].sum() / 1000.0
    return spindle_s / total_nrem_s


def compare_rules(spindles, ripples):
    """Prints how much the two rules differ."""
    out = {}
    for rule in ["peak_in_spindle", "any_overlap"]:
        t = count_overlap(spindles, ripples, rule=rule)
        out[rule] = (t["n_coupled"].sum(), t["n_ripple"].sum())
    (a, n), (b, _) = out["peak_in_spindle"], out["any_overlap"]
    print(f"  peak_in_spindle : {a:>8,} of {n:,} ripples ({100 * a / n:.2f}%)")
    print(f"  any_overlap     : {b:>8,} of {n:,} ripples ({100 * b / n:.2f}%)")
    print(f"  difference      : {b - a:,} ripples ({100 * (b - a) / n:.2f} pp)")
    return out


# surrogate: circular shift, same idea as in surrogate.py

REC_KEYS = ["rat", "date", "trial"]


def _shift_within_recording(sg, col_lo, col_hi, rng, min_shift_ms):
    """Circular shift on the concatenated NREM timeline of one recording.

    Shifts an interval event (keeping its duration) or a point event.
    Returns None if the recording is too short to shift.
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
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = float(lengths.sum())
    if total <= 2 * min_shift_ms:
        return None

    pos = np.clip(
        np.searchsorted(idx, sg["nrem_bout_index"].to_numpy()), 0, len(idx) - 1
    )
    dur = (
        sg[col_hi].to_numpy(dtype=float) - sg[col_lo].to_numpy(dtype=float)
        if col_hi != col_lo
        else np.zeros(len(sg))
    )
    offset = starts[pos] + (sg[col_lo].to_numpy(dtype=float) - lo[pos])

    shift = rng.uniform(min_shift_ms, total - min_shift_ms)
    new_off = np.mod(offset + shift, total)

    j = np.clip(np.searchsorted(starts, new_off, side="right") - 1, 0, len(starts) - 1)

    out = sg.copy()
    out[col_lo] = lo[j] + (new_off - starts[j])
    if col_hi != col_lo:
        out[col_hi] = out[col_lo] + dur
    out["nrem_bout_index"] = idx[j]
    return out


def surrogate_overlap(
    spindles,
    ripples,
    rule="peak_in_spindle",
    n_iter=200,
    seed=0,
    min_shift_ms=1000,
    verbose=True,
    checkpoint_every=None,
    checkpoint_fn=None,
    resume_from=None,
):
    """Null distribution for the overlap count (circular shift of the ripples).

    checkpoint_every / checkpoint_fn: checkpoint_fn(partial) is called every
        `checkpoint_every` iterations with the stats so far.
    resume_from: earlier result or checkpoint (same seed); iterations reused.

    Returns dict with real, null, mu, sd, z, p, ratio, n_ripple, n_done and a
    "spindle_level" dict with the same fields for spindles with a ripple.
    """
    real_t = count_overlap(spindles, ripples, rule=rule)
    real = int(real_t["n_coupled"].sum())
    real_sp = int(real_t["n_spindle_coupled"].sum())
    n_rip = int(real_t["n_ripple"].sum())
    n_spn = int(real_t["n_spindle"].sum())

    rp = ripples.dropna(subset=BOUT_KEYS + ["ripple_peak_index"])
    col_lo = "ripple_peak_index"
    col_hi = "ripple_end_index" if rule == "any_overlap" else "ripple_peak_index"
    if rule == "any_overlap":
        col_lo = "ripple_start_index"
        rp = rp.dropna(subset=["ripple_start_index", "ripple_end_index"])

    null = np.zeros(n_iter)
    null_sp = np.zeros(n_iter)
    start = 0
    rng = np.random.default_rng(seed)

    if resume_from is not None:
        prev_n = min(int(resume_from.get("n_done", 0)), n_iter)
        null[:prev_n] = resume_from["null"][:prev_n]
        null_sp[:prev_n] = resume_from["spindle_level"]["null"][:prev_n]
        start = prev_n
        if verbose:
            print(f"  resuming from checkpoint: {start}/{n_iter} already done")
        # advance the RNG past the iterations already done
        for _ in range(start):
            for _, sg_ in rp.groupby(REC_KEYS):
                _shift_within_recording(sg_, col_lo, col_hi, rng, min_shift_ms)

    def _stats(obs, dist, n_ok):
        mu, sd = np.nanmean(dist[:n_ok]), np.nanstd(dist[:n_ok])
        n_finite = np.sum(~np.isnan(dist[:n_ok]))
        return {
            "real": int(obs),
            "null": dist,
            "mu": float(mu),
            "sd": float(sd),
            "z": float((obs - mu) / sd) if sd > 0 else np.nan,
            "p": float((1 + np.nansum(dist[:n_ok] >= obs)) / (1 + n_finite)),
            "ratio": float(obs / mu) if mu > 0 else np.nan,
            "n_done": n_ok,
        }

    for i in range(start, n_iter):
        parts = []
        for _, sg_ in rp.groupby(REC_KEYS):
            sh = _shift_within_recording(sg_, col_lo, col_hi, rng, min_shift_ms)
            if sh is not None:
                parts.append(sh)
        if not parts:
            null[i] = null_sp[i] = np.nan
        else:
            shifted = pd.concat(parts, ignore_index=True)
            t = count_overlap(spindles, shifted, rule=rule)
            null[i] = t["n_coupled"].sum()
            null_sp[i] = t["n_spindle_coupled"].sum()

        done = i + 1
        if verbose and done % 50 == 0:
            print(f"    {done}/{n_iter}")
        if checkpoint_every and checkpoint_fn and done % checkpoint_every == 0:
            out = _stats(real, null, done)
            out["n_ripple"] = n_rip
            out["n_spindle"] = n_spn
            out["spindle_level"] = _stats(real_sp, null_sp, done)
            checkpoint_fn(out)

    out = _stats(real, null, n_iter)  # ripple-level
    out["n_ripple"] = n_rip
    out["n_spindle"] = n_spn
    out["spindle_level"] = _stats(real_sp, null_sp, n_iter)  # Navarro Lobato's unit
    return out


# self tests: python spindle_ripple.py


def _synth(
    n_bouts=40, bout_ms=60_000, n_spindle=25, n_ripple=40, coupled_frac=0.0, seed=0
):
    """Spindles of ~1 s, ripples independent or planted inside them."""
    rng = np.random.default_rng(seed)
    S, R = [], []
    for b in range(n_bouts):
        s_lo = np.sort(rng.uniform(0, bout_ms - 1200, n_spindle))
        s_dur = rng.uniform(500, 1500, n_spindle)
        r_pk = rng.uniform(0, bout_ms, n_ripple)

        if coupled_frac > 0:
            k = int(coupled_frac * n_ripple)
            pick = rng.choice(n_spindle, k, replace=True)
            r_pk[:k] = s_lo[pick] + rng.uniform(0, 1, k) * s_dur[pick]

        S.append(
            pd.DataFrame(
                {
                    "rat": 1,
                    "date": "x",
                    "trial": 1,
                    "nrem_bout_index": b,
                    "nrem_bout_start_index": 0.0,
                    "nrem_bout_end_index": float(bout_ms),
                    "spindle_start_index": s_lo,
                    "spindle_end_index": s_lo + s_dur,
                }
            )
        )
        R.append(
            pd.DataFrame(
                {
                    "rat": 1,
                    "date": "x",
                    "trial": 1,
                    "nrem_bout_index": b,
                    "nrem_bout_start_index": 0.0,
                    "nrem_bout_end_index": float(bout_ms),
                    "ripple_peak_index": np.sort(r_pk),
                    "ripple_start_index": np.sort(r_pk) - 30,
                    "ripple_end_index": np.sort(r_pk) + 30,
                }
            )
        )
    return pd.concat(S, ignore_index=True), pd.concat(R, ignore_index=True)


if __name__ == "__main__":
    print("TEST 1 -- hand-checkable case")
    sp = pd.DataFrame(
        {
            "rat": 1,
            "date": "x",
            "trial": 1,
            "nrem_bout_index": 1,
            "spindle_start_index": [1000.0, 5000.0],
            "spindle_end_index": [2000.0, 6000.0],
        }
    )
    rp = pd.DataFrame(
        {
            "rat": 1,
            "date": "x",
            "trial": 1,
            "nrem_bout_index": 1,
            "ripple_peak_index": [1500.0, 3000.0, 5500.0, 5800.0],
            "ripple_start_index": [1470.0, 2970.0, 5470.0, 5770.0],
            "ripple_end_index": [1530.0, 3030.0, 5530.0, 5830.0],
        }
    )
    t = count_overlap(sp, rp)
    print(f"  4 ripples, 2 inside spindles -> n_coupled = {t['n_coupled'][0]} (want 3)")

    print("\nTEST 2 -- no real coupling: observed should match the null")
    sp, rp = _synth(coupled_frac=0.0, seed=1)
    r = surrogate_overlap(sp, rp, n_iter=100, seed=1, verbose=False)
    print(
        f"  observed {r['real']} of {r['n_ripple']} ({100 * r['real'] / r['n_ripple']:.1f}%) "
        f"| null {r['mu']:.0f} | ratio {r['ratio']:.3f} | z {r['z']:+.2f} "
        f"| p {r['p']:.3f}   (want ratio ~1)"
    )

    print("\nTEST 3 -- 30% of ripples planted inside spindles")
    sp, rp = _synth(coupled_frac=0.30, seed=2)
    r = surrogate_overlap(sp, rp, n_iter=100, seed=2, verbose=False)
    print(
        f"  observed {r['real']} of {r['n_ripple']} ({100 * r['real'] / r['n_ripple']:.1f}%) "
        f"| null {r['mu']:.0f} | ratio {r['ratio']:.3f} | z {r['z']:+.1f} "
        f"| p {r['p']:.3f}   (want ratio >> 1)"
    )

    print("\nTEST 4 -- does the rule choice matter?")
    sp, rp = _synth(coupled_frac=0.15, seed=3)
    compare_rules(sp, rp)
