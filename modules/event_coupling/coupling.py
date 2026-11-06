# coupling.py
# Directed peak-to-peak sequences (e.g. delta -> ripple) within the same NREM bout.
# 1000 Hz, so 1 sample = 1 ms.
#
# Windows (Maingret 2016, Navarro-Lobato 2023):
#   D -> R : 50-400 ms      R -> D : 50-250 ms      R -> S : 2-1000 ms
import numpy as np
import pandas as pd

# columns that define one NREM bout
BOUT_KEYS = ["rat", "date", "trial", "nrem_bout_index"]

WINDOWS_MS = {
    ("delta", "ripple"): (50, 400),
    ("ripple", "delta"): (50, 250),
    ("ripple", "spindle"): (2, 1000),
}


def _peak_col(wave):
    return f"{wave}_peak_index"


def count_sequences(
    first_df, second_df, first_wave, second_wave, window_ms=None, second_area=None
):
    """Count directed first->second sequences per bout.

    For each first event, the number of second events in the same bout with a
    peak window_lo..window_hi ms later. One row per bout.

    Parameters
    ----------
    first_df, second_df : event tables (already condition-labelled)
    first_wave, second_wave : "delta" / "ripple" / "spindle"
    window_ms : (lo, hi) in ms; defaults to the published window for the pair
    second_area : optional area filter for the second wave
    """
    if window_ms is None:
        window_ms = WINDOWS_MS[(first_wave, second_wave)]
    lo, hi = window_ms

    fp, sp = _peak_col(first_wave), _peak_col(second_wave)

    f = first_df.dropna(subset=BOUT_KEYS + [fp]).copy()
    s = second_df.dropna(subset=BOUT_KEYS + [sp]).copy()
    if second_area is not None and "area" in s.columns:
        s = s[s["area"] == second_area]

    # only the needed columns
    f = f[BOUT_KEYS + [fp]].rename(columns={fp: "first_peak"})
    s = s[BOUT_KEYS + [sp]].rename(columns={sp: "second_peak"})

    rows = []
    # loop over the bouts of the first table
    s_groups = dict(tuple(s.groupby(BOUT_KEYS)))
    for key, fg in f.groupby(BOUT_KEYS):
        sg = s_groups.get(key)
        n_first = len(fg)
        if sg is None or sg.empty:
            rows.append((*key, n_first, 0, 0))
            continue
        n_second = len(sg)
        # second peaks in (peak+lo, peak+hi)
        first_peaks = fg["first_peak"].to_numpy()
        second_peaks = np.sort(sg["second_peak"].to_numpy())
        # vectorised: for each first peak, count second peaks in the window
        lo_idx = np.searchsorted(second_peaks, first_peaks + lo, side="left")
        hi_idx = np.searchsorted(second_peaks, first_peaks + hi, side="right")
        n_seq = int((hi_idx - lo_idx).sum())
        rows.append((*key, n_first, n_second, n_seq))

    out = pd.DataFrame(
        rows, columns=BOUT_KEYS + [f"n_{first_wave}", f"n_{second_wave}", "n_sequences"]
    )
    return out
