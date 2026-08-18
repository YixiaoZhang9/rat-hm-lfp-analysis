import config
import numpy as np

# Adjust imports according to your module setup
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal
from modules.iaaft import surrogates as iaaft_surrogates


class TaskFailure(Exception):
    """Raised with a short, categorical reason so failures can be tallied."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def get_nrem_intervals(states_array):
    nrem_mask = (states_array == 3).astype(int)
    diff = np.diff(np.concatenate(([0], nrem_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return np.empty((0, 2))
    return np.column_stack((starts, ends))


def get_random_nrem_segment(raw_signal, nrem_intervals, fs, target_sec):
    chunks = [raw_signal[int(s * fs) : int(e * fs)] for s, e in nrem_intervals]
    if not chunks:
        return np.array([])

    pooled_signal = np.concatenate(chunks)
    target_samples = target_sec * fs

    if len(pooled_signal) <= target_samples:
        return pooled_signal

    start_idx = np.random.randint(0, len(pooled_signal) - target_samples)
    return pooled_signal[start_idx : start_idx + target_samples]


def calculate_optimal_threshold(raw_signal, states_array):
    nrem_intervals = get_nrem_intervals(states_array)
    if nrem_intervals.shape[0] == 0:
        raise TaskFailure("no NREM epochs in scoring file")

    random_real_raw = get_random_nrem_segment(raw_signal, nrem_intervals, config.FS, config.SEGMENT_SEC)
    if random_real_raw.size == 0:
        raise TaskFailure("empty NREM segment after pooling")

    filtered = bandpass_filter(random_real_raw, lowcut=0.1, highcut=100, fs=config.FS)
    pooled_128 = downsampling(filtered, config.FS, config.TARGET_FS)

    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, config.TARGET_FS, n_jobs=-1, verbose=False)
    valid_r_real = r_real[~np.isnan(f_real)]

    if len(valid_r_real) == 0:
        raise TaskFailure("AR fit on real data returned no valid windows")

    surrs = iaaft_surrogates(pooled_128, ns=config.N_SURROGATES, verbose=False)
    all_surr_r = []

    for surrogate in surrs:
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, config.TARGET_FS, n_jobs=-1, verbose=False)
        all_surr_r.append(r_surr[~np.isnan(f_surr)])

    all_surr_r = np.concatenate(all_surr_r) if all_surr_r else np.array([])
    if all_surr_r.size == 0:
        raise TaskFailure("AR fit on all surrogates returned no valid windows")

    # Vectorized threshold evaluation
    thresholds = np.arange(0.70, 0.90, 0.01)

    real_counts = np.array([np.sum(valid_r_real >= t) for t in thresholds])
    surr_counts = np.array([np.sum(all_surr_r >= t) for t in thresholds]) / config.N_SURROGATES

    valid_mask = real_counts > 0
    if not np.any(valid_mask):
        raise TaskFailure("no threshold candidate satisfied count_real > 0")

    pct_diffs = ((real_counts[valid_mask] - surr_counts[valid_mask]) / real_counts[valid_mask]) * 100
    closest_idx = np.argmin(np.abs(pct_diffs - config.TARGET_PCT_DIFF))

    return thresholds[valid_mask][closest_idx]
