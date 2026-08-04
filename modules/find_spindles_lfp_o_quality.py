import contextlib
import logging
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from spectrum import arburg
from tqdm import tqdm

from modules.ephys_preprocessing import bandpass_filter, downsampling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report progress into a tqdm bar."""
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
    a, noise, _ = arburg(window, order=ar_order)
    poles = np.roots(np.r_[1, -a])
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


def fit_ar_on_prepared_signal(
    signal,
    target_fs=128,
    ar_order=8,
    window_sec=1.0,
    spindle_band=(10, 15),
    n_jobs=-1,
    verbose=True,
):
    """
    Fit AR(8) models across overlapping windows of a signal that is ALREADY
    filtered and downsampled (i.e. skips steps 1 & 2 of the pipeline).

    Returns
    -------
    r_timeseries, f_timeseries : np.ndarray
    """
    window_samples = int(window_sec * target_fs)

    if len(signal) < window_samples:
        return np.array([]), np.array([])

    total_windows = len(signal) - window_samples + 1
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_samples)

    if verbose:
        with tqdm_joblib(tqdm(total=total_windows, desc="AR fitting", unit="win")):
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_fit_window)(windows[start], ar_order, target_fs, spindle_band)
                for start in range(total_windows)
            )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_window)(windows[start], ar_order, target_fs, spindle_band)
            for start in range(total_windows)
        )

    r_timeseries = np.array([r for r, f in results])
    f_timeseries = np.array([f for r, f in results])
    return r_timeseries, f_timeseries


def compute_r_f_timeseries(
    raw_signal,
    fs,
    target_fs=128,
    ar_order=8,
    window_sec=1.0,
    spindle_band=(10, 15),
    n_jobs=-1,
    verbose=True,
):
    """
    Full pipeline: band-pass filter (0.1-100 Hz) -> downsample to target_fs
    -> AR(8) fitting. Used both by the detector (find_spindles_lfp) and by
    the threshold calibration script.
    """
    filtered_signal = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=fs)
    signal = downsampling(filtered_signal, fs, target_fs)
    return fit_ar_on_prepared_signal(
        signal, target_fs, ar_order, window_sec, spindle_band, n_jobs, verbose
    )


def find_spindles_lfp(
    raw_signal,
    fs,
    target_fs=128,
    ar_order=8,
    window_sec=1.0,
    upper_threshold=0.92,
    lower_threshold=0.90,
    spindle_band=(10, 15),
    n_jobs=-1,
):
    t_start = time.time()
    logging.info(f"Starting spindle detection. Input signal length: {len(raw_signal)}, original fs: {fs}")

    r_timeseries, f_timeseries = compute_r_f_timeseries(
        raw_signal=raw_signal,
        fs=fs,
        target_fs=target_fs,
        ar_order=ar_order,
        window_sec=window_sec,
        spindle_band=spindle_band,
        n_jobs=n_jobs,
        verbose=True,
    )

    window_samples = int(window_sec * target_fs)

    if len(r_timeseries) == 0:
        logging.warning("Signal too short for window.")
        return np.empty((0, 6))

    # Detect events using upper/lower thresholds
    logging.info(f"Scanning r-timeseries for spindle events "
                 f"(upper={upper_threshold}, lower={lower_threshold})...")
    t0 = time.time()
    events = []
    in_event = False
    start_idx = None

    for i, r in enumerate(r_timeseries):
        if not in_event and r >= upper_threshold:
            in_event = True
            start_idx = i

        elif in_event and r < lower_threshold:
            end_idx = i - 1

            event_slice = slice(start_idx, end_idx + 1)
            peak_relative = np.argmax(r_timeseries[event_slice])
            peak_idx = start_idx + peak_relative

            max_r = r_timeseries[peak_idx]
            peak_frequency = f_timeseries[peak_idx]

            start_time = start_idx / target_fs
            end_time = (end_idx + window_samples) / target_fs
            duration = end_time - start_time

            events.append([
                start_time,
                peak_idx / target_fs,
                end_time,
                duration,
                max_r,
                peak_frequency,
            ])

            in_event = False

    if in_event:
        end_idx = len(r_timeseries) - 1

        event_slice = slice(start_idx, end_idx + 1)
        peak_relative = np.argmax(r_timeseries[event_slice])
        peak_idx = start_idx + peak_relative

        max_r = r_timeseries[peak_idx]
        peak_frequency = f_timeseries[peak_idx]

        start_time = start_idx / target_fs
        end_time = (end_idx + window_samples) / target_fs
        duration = end_time - start_time

        events.append([
            start_time,
            peak_idx / target_fs,
            end_time,
            duration,
            max_r,
            peak_frequency,
        ])

    logging.info(f"Event scan complete in {time.time() - t0:.2f}s. Found {len(events)} events.")

    valid_r = r_timeseries[~np.isnan(f_timeseries)]
    if valid_r.size > 0:
        logging.info("--- Global R Statistics (in spindle band) ---")
        logging.info(f"Min: {valid_r.min():.3f} | Max: {valid_r.max():.3f} | Avg: {valid_r.mean():.3f} | Std: {valid_r.std():.3f}")

    logging.info(f"Total detection time: {time.time() - t_start:.2f}s.")

    return np.asarray(events)
