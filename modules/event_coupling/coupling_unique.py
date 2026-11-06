"""
count_sequences_unique: Yixiao's counting rule (30 July 2026).

    "Delta can be counted more than once in a sequence, but ripples can only
     be counted once."

Each ripple is coupled or not and counted once (count_sequences counts pairs).

Both directions:
  delta -> ripple : unique ripples with at least one delta 50-400 ms BEFORE
  ripple -> delta : unique ripples with at least one delta 50-250 ms AFTER
"""

import numpy as np
import pandas as pd

BOUT_KEYS = ["rat", "date", "trial", "nrem_bout_index"]

WINDOWS_MS = {
    ("delta", "ripple"): (50, 400),
    ("ripple", "delta"): (50, 250),
    ("ripple", "spindle"): (2, 1000),
}


def count_sequences_unique(
    first_df, second_df, first_wave, second_wave, window_ms=None
):
    """Count sequences with each ripple used at most once.

    Returns one row per bout:
        n_<first_wave>, n_<second_wave>  : events in the bout
        n_sequences                      : unique ripples involved in a sequence
        n_pairs                          : old pair count, kept for comparison

    """
    if window_ms is None:
        window_ms = WINDOWS_MS[(first_wave, second_wave)]
    lo, hi = window_ms

    fp, sp = f"{first_wave}_peak_index", f"{second_wave}_peak_index"

    f = first_df.dropna(subset=BOUT_KEYS + [fp])[BOUT_KEYS + [fp]]
    s = second_df.dropna(subset=BOUT_KEYS + [sp])[BOUT_KEYS + [sp]]

    s_groups = dict(tuple(s.groupby(BOUT_KEYS)))

    rows = []
    for key, fg in f.groupby(BOUT_KEYS):
        sg = s_groups.get(key)
        n_first = len(fg)
        if sg is None or sg.empty:
            rows.append((*key, n_first, 0, 0, 0))
            continue

        first_peaks = np.sort(fg[fp].to_numpy())
        second_peaks = np.sort(sg[sp].to_numpy())
        n_second = len(second_peaks)

        # for each first event, the slice of second events inside its window
        lo_idx = np.searchsorted(second_peaks, first_peaks + lo, side="left")
        hi_idx = np.searchsorted(second_peaks, first_peaks + hi, side="right")

        n_pairs = int((hi_idx - lo_idx).sum())

        if first_wave == "ripple":
            # ripple first: counted if its window has a partner
            n_unique = int((hi_idx > lo_idx).sum())
        else:
            # ripple second: ripples inside at least one delta window
            covered = np.zeros(n_second, dtype=bool)
            for a, b in zip(lo_idx, hi_idx):
                if b > a:
                    covered[a:b] = True
            n_unique = int(covered.sum())

        rows.append((*key, n_first, n_second, n_unique, n_pairs))

    return pd.DataFrame(
        rows,
        columns=BOUT_KEYS
        + [f"n_{first_wave}", f"n_{second_wave}", "n_sequences", "n_pairs"],
    )


def expected_by_chance(n_ripples, other_rate_per_s, window_ms):
    """Unique ripples expected to be 'coupled' if timing were random.

    A ripple counts as coupled if at least one partner event falls in its
    window. For randomly-timed partners at rate r, the number in a window of
    width w is Poisson with mean r*w, so

        P(at least one) = 1 - exp(-r * w)

    Depends on partner density, so compute per area.
    """
    lo, hi = window_ms
    width_s = (hi - lo) / 1000.0
    p_coupled = 1.0 - np.exp(-other_rate_per_s * width_s)
    return n_ripples * p_coupled, p_coupled
