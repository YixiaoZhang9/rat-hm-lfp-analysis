import contextlib
import logging
import time

import joblib
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm

from modules.ephys_preprocessing import bandpass_filter, downsampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _fit_window(window, ar_order, target_fs, spindle_band):
    """Fit AR model to a single window and return (best_r, best_freq)."""
    try:
        a, _ = sm.regression.linear_model.burg(window, order=ar_order, demean=False)
        poles = np.roots(np.r_[1, -a])
        poles = poles[np.imag(poles) > 0]
        frequencies = np.angle(poles) * target_fs / (2 * np.pi)
        r_values = np.abs(poles)

        mask = (frequencies >= spindle_band[0]) & (frequencies <= spindle_band[1])
        frequencies = frequencies[mask]
        r_values = r_values[mask]

        if len(r_values) > 0:
            best = np.argmax(r_values)
            return float(r_values[best]), float(frequencies[best])
    except Exception:
        pass
    return 0.0, np.nan


def _fit_windows_at_starts(signal, starts, window_samples, ar_order, target_fs, spindle_band, n_jobs, verbose, desc):
    """Fit AR windows starting at arbitrary sample indices."""
    if len(starts) == 0:
        return np.array([]), np.array([])

    if verbose:
        with tqdm_joblib(tqdm(total=len(starts), desc=desc, unit="win")):
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_fit_window)(signal[s:s + window_samples], ar_order, target_fs, spindle_band)
                for s in starts
            )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_window)(signal[s:s + window_samples], ar_order, target_fs, spindle_band)
            for s in starts
        )

    r_vals = np.array([r for r, f in results])
    f_vals = np.array([f for r, f in results])
    return r_vals, f_vals


def fit_ar_on_prepared_signal(
    signal, target_fs=128, ar_order=8, window_sec=1.0,
    spindle_band=(10, 15), n_jobs=-1, verbose=True,
):
    window_samples = int(window_sec * target_fs)
    if len(signal) < window_samples:
        return np.array([]), np.array([])

    total_windows = len(signal) - window_samples + 1
    starts = np.arange(total_windows)
    return _fit_windows_at_starts(
        signal, starts, window_samples, ar_order, target_fs, spindle_band,
        n_jobs, verbose, desc="AR fitting (full-res)"
    )


def _detect_events_in_region(r, f, sample_starts, target_fs, window_samples, upper_threshold, lower_threshold):
    """Detect event intervals using hysteresis thresholding."""
    events = []
    in_event = False
    start_i = None
    last_above_rb_i = None

    for i, r_val in enumerate(r):
        if not in_event and r_val >= upper_threshold:
            in_event = True
            start_i = i
            last_above_rb_i = i
        elif in_event:
            if r_val >= upper_threshold:
                last_above_rb_i = i
            if r_val < lower_threshold:
                _finalize_event(events, r, f, sample_starts, target_fs, window_samples, start_i, last_above_rb_i)
                in_event = False

    if in_event:
        _finalize_event(events, r, f, sample_starts, target_fs, window_samples, start_i, last_above_rb_i)

    return events


def _finalize_event(events, r, f, sample_starts, target_fs, window_samples, start_i, end_i):
    peak_relative = np.argmax(r[start_i:end_i + 1])
    peak_i = start_i + peak_relative

    max_r = r[peak_i]
    peak_frequency = f[peak_i]

    start_time = sample_starts[start_i] / target_fs
    end_time = (sample_starts[end_i] + window_samples) / target_fs
    peak_time = sample_starts[peak_i] / target_fs
    duration = end_time - start_time

    events.append([start_time, peak_time, end_time, duration, max_r, peak_frequency])


def merge_overlapping_events(events_list, min_gap_sec=0.5):
    """
    Merge overlapping or closely consecutive detected intervals.

    Parameters
    ----------
    events_list : list of [start, peak, end, duration, max_r, peak_freq]
    min_gap_sec : float
        Intervals separated by less than or equal to this duration will be merged.
    """
    if not events_list:
        return np.empty((0, 6))

    # Sort primarily by start time
    events = sorted(events_list, key=lambda x: x[0])
    merged = []

    curr_start, curr_peak, curr_end, _, curr_max_r, curr_freq = events[0]

    for next_ev in events[1:]:
        n_start, n_peak, n_end, _, n_max_r, n_freq = next_ev

        # Check if events overlap or gap is below min_gap_sec
        if n_start <= curr_end + min_gap_sec:
            # Merge intervals
            curr_end = max(curr_end, n_end)
            if n_max_r > curr_max_r:
                curr_max_r = n_max_r
                curr_peak = n_peak
                curr_freq = n_freq
        else:
            merged.append([
                curr_start,
                curr_peak,
                curr_end,
                curr_end - curr_start,
                curr_max_r,
                curr_freq
            ])
            curr_start, curr_peak, curr_end, _, curr_max_r, curr_freq = next_ev

    merged.append([
        curr_start,
        curr_peak,
        curr_end,
        curr_end - curr_start,
        curr_max_r,
        curr_freq
    ])

    return np.asarray(merged)


def find_spindles_lfp(
    raw_signal, fs, target_fs=128, ar_order=8, window_sec=1.0,
    upper_threshold=0.75, spindle_band=(10, 15), min_gap_sec=0.5,
    n_jobs=-1,
):
    lower_threshold = upper_threshold - 0.02
    t_start = time.time()
    logging.info(f"Starting spindle detection, Input signal length: {len(raw_signal)}")

    filtered_signal = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=fs)
    signal = downsampling(filtered_signal, fs, target_fs)
    window_samples = int(window_sec * target_fs)

    raw_events = []
    r_timeseries, f_timeseries = fit_ar_on_prepared_signal(
        signal, target_fs=target_fs, ar_order=ar_order, window_sec=window_sec,
        spindle_band=spindle_band, n_jobs=n_jobs, verbose=True,
    )
    if len(r_timeseries) == 0:
        logging.warning("Signal too short for window.")
        return np.empty((0, 6))
    raw_events = _detect_events_in_region(
        r_timeseries, f_timeseries, np.arange(len(r_timeseries)),
        target_fs, window_samples, upper_threshold, lower_threshold,
    )

    # Post-processing: consolidate duplicate sliding windows
    final_events = merge_overlapping_events(raw_events, min_gap_sec=min_gap_sec)

    logging.info(
        f"Detection complete in {time.time() - t_start:.2f}s. "
        f"Raw detections: {len(raw_events)} -> Merged events: {len(final_events)}"
    )

    return final_events
