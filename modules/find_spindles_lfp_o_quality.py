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
    a, _ = sm.regression.linear_model.burg(window, order=ar_order, demean=False)
    poles = np.roots(np.r_[1, a])
    poles = poles[np.imag(poles) > 0]
    frequencies = np.angle(poles) * target_fs / (2 * np.pi)
    r_values = np.abs(poles)

    mask = (frequencies >= spindle_band[0]) & (frequencies <= spindle_band[1])
    frequencies = frequencies[mask]
    r_values = r_values[mask]

    if len(r_values) > 0:
        best = np.argmax(r_values)
        return r_values[best], frequencies[best]
    return 0.0, np.nan


def _fit_windows_at_starts(signal, starts, window_samples, ar_order, target_fs, spindle_band, n_jobs, verbose, desc):
    """Fit AR windows starting at arbitrary sample indices (used for both coarse & fine passes)."""
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
    """
    Full-resolution method: 1-sample-shift AR fitting across the ENTIRE
    signal. Slow but exhaustive -- required for calibration/significance
    testing where an unbiased global max r is needed (see discussion:
    the two-pass method can systematically under/over-estimate peaks
    near coarse-window boundaries).
    """
    window_samples = int(window_sec * target_fs)
    if len(signal) < window_samples:
        return np.array([]), np.array([])

    total_windows = len(signal) - window_samples + 1
    starts = np.arange(total_windows)
    return _fit_windows_at_starts(
        signal, starts, window_samples, ar_order, target_fs, spindle_band,
        n_jobs, verbose, desc="AR fitting (full-res)"
    )


def fit_ar_two_pass(
    signal, target_fs=128, ar_order=8, window_sec=1.0,
    spindle_band=(10, 15), ra=0.90, fine_step_frac=1 / 16,
    n_jobs=-1, verbose=True,
):
    """
    Paper's two-pass strategy (Olbrich & Achermann 2005):
    1. Coarse scan: non-overlapping 1-s windows across the WHOLE signal.
    2. Wherever coarse r >= ra, flag that segment AND the previous one.
    3. Fine scan (1/16 s step) ONLY within flagged regions.

    Returns
    -------
    regions : list of dicts, each with keys:
        'start_sample', 'r', 'f', 'sample_starts' (fine-grained, absolute sample idx)
    """
    window_samples = int(window_sec * target_fs)
    fine_step = max(1, int(round(window_samples * fine_step_frac)))

    if len(signal) < window_samples:
        return []

    n_coarse = (len(signal) - window_samples) // window_samples + 1
    coarse_starts = np.arange(n_coarse) * window_samples
    r_coarse, f_coarse = _fit_windows_at_starts(
        signal, coarse_starts, window_samples, ar_order, target_fs, spindle_band,
        n_jobs, verbose, desc="Coarse AR scan"
    )

    flagged = np.zeros(n_coarse, dtype=bool)
    hits = np.where(r_coarse >= ra)[0]
    flagged[hits] = True
    flagged[np.clip(hits - 1, 0, n_coarse - 1)] = True

    if not flagged.any():
        logging.info("No candidate regions found in coarse scan (nothing exceeded ra).")
        return []

    regions = []
    i = 0
    while i < n_coarse:
        if flagged[i]:
            j = i
            while j + 1 < n_coarse and flagged[j + 1]:
                j += 1
            region_start = coarse_starts[i]
            region_end = coarse_starts[j] + window_samples  # exclusive
            regions.append((region_start, region_end))
            i = j + 1
        else:
            i += 1

    logging.info(f"Coarse scan flagged {len(regions)} region(s) for fine-grained analysis "
                 f"({sum(e - s for s, e in regions) / target_fs:.1f}s of {len(signal) / target_fs:.1f}s total).")

    fine_regions = []
    for region_start, region_end in regions:
        last_valid_start = region_end - window_samples
        if last_valid_start < region_start:
            continue
        fine_starts = np.arange(region_start, last_valid_start + 1, fine_step)
        r_fine, f_fine = _fit_windows_at_starts(
            signal, fine_starts, window_samples, ar_order, target_fs, spindle_band,
            n_jobs, verbose=False, desc="Fine AR scan"
        )
        fine_regions.append({
            "start_sample": region_start,
            "r": r_fine,
            "f": f_fine,
            "sample_starts": fine_starts,
        })

    return fine_regions


def _detect_events_in_region(r, f, sample_starts, target_fs, window_samples, upper_threshold, lower_threshold):
    """
    Correct t1/t2 event logic per Olbrich & Achermann (2005):
    t1 = first upward crossing of rb.
    t2 = LAST time r was >= rb, before it falls below ra.
    """
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
                _finalize_event(events, r, f, sample_starts, target_fs, window_samples,
                                 start_i, last_above_rb_i)
                in_event = False

    if in_event:
        _finalize_event(events, r, f, sample_starts, target_fs, window_samples,
                         start_i, last_above_rb_i)

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


def _detect_events_full_res(r_timeseries, f_timeseries, target_fs, window_samples, upper_threshold, lower_threshold):
    """Event detection over a single, contiguous, full-resolution r/f timeseries
    (i.e. sample_starts are just 0..N-1)."""
    sample_starts = np.arange(len(r_timeseries))
    return _detect_events_in_region(
        r_timeseries, f_timeseries, sample_starts, target_fs, window_samples,
        upper_threshold, lower_threshold,
    )


def compute_r_f_timeseries(
    raw_signal, fs, target_fs=128, ar_order=8, window_sec=1.0,
    spindle_band=(10, 15), n_jobs=-1, verbose=True,
):
    """Full pipeline (unchanged): filter -> downsample -> full-res AR fitting."""
    filtered_signal = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=fs)
    signal = downsampling(filtered_signal, fs, target_fs)
    return fit_ar_on_prepared_signal(
        signal, target_fs, ar_order, window_sec, spindle_band, n_jobs, verbose
    )


def find_spindles_lfp(
    raw_signal, fs, target_fs=128, ar_order=8, window_sec=1.0,
    upper_threshold=0.75, spindle_band=(10, 15), n_jobs=-1,
    method="two_pass",
):
    """
    Spindle detection with correct t1/t2 event boundary logic.

    Parameters
    ----------
    method : {"two_pass", "full"}
        "two_pass" (default): paper's coarse/fine strategy (Olbrich &
            Achermann 2005). Fast, recommended for standard detection runs.
        "full": exhaustive 1-sample-shift AR fitting across the entire
            signal. Slow, but avoids any risk of missing peaks near
            coarse-window boundaries. Recommended for validation runs or
            anywhere exact global-max fidelity matters (e.g. calibration-
            style analyses run through this function).
    """
    if method not in ("two_pass", "full"):
        raise ValueError(f"method must be 'two_pass' or 'full', got {method!r}")

    lower_threshold = upper_threshold - 0.02
    t_start = time.time()
    logging.info(f"Starting spindle detection (method={method}). "
                 f"Input signal length: {len(raw_signal)}, original fs: {fs}")

    filtered_signal = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=fs)
    signal = downsampling(filtered_signal, fs, target_fs)
    window_samples = int(window_sec * target_fs)

    if method == "two_pass":
        fine_regions = fit_ar_two_pass(
            signal, target_fs=target_fs, ar_order=ar_order, window_sec=window_sec,
            spindle_band=spindle_band, ra=lower_threshold, n_jobs=n_jobs, verbose=True,
        )
        all_events = []
        for region in fine_regions:
            events = _detect_events_in_region(
                region["r"], region["f"], region["sample_starts"],
                target_fs, window_samples, upper_threshold, lower_threshold,
            )
            all_events.extend(events)

    else:  # method == "full"
        r_timeseries, f_timeseries = fit_ar_on_prepared_signal(
            signal, target_fs=target_fs, ar_order=ar_order, window_sec=window_sec,
            spindle_band=spindle_band, n_jobs=n_jobs, verbose=True,
        )
        if len(r_timeseries) == 0:
            logging.warning("Signal too short for window.")
            return np.empty((0, 6))
        all_events = _detect_events_full_res(
            r_timeseries, f_timeseries, target_fs, window_samples,
            upper_threshold, lower_threshold,
        )

    logging.info(f"Detection complete in {time.time() - t_start:.2f}s. Found {len(all_events)} events.")

    return np.asarray(all_events) if all_events else np.empty((0, 6))
