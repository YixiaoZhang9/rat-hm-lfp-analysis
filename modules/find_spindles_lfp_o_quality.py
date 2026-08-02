import numpy as np
from spectrum import arburg
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import joblib
import logging
import time

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

    # 1. Band-pass filter: 0.1–100 Hz
    logging.info("Band-pass filtering signal from 0.1–100 Hz...")
    t0 = time.time()
    filtered_signal = bandpass_filter(
        raw_signal,
        lowcut=0.1,
        highcut=100,
        fs=fs,
    )
    logging.info(f"Band-pass filtering complete in {time.time() - t0:.2f}s.")

    # 2. Resample to 128 Hz
    logging.info(f"Downsampling to {target_fs} Hz...")
    t0 = time.time()
    signal = downsampling(filtered_signal, fs, target_fs)
    logging.info(f"Downsampling complete in {time.time() - t0:.2f}s. New length: {len(signal)}")

    window_samples = int(window_sec * target_fs)

    if len(signal) < window_samples:
        logging.warning("Signal too short for window.")
        return np.empty((0, 6))

    # 2. Fit AR(8) to overlapping 1-second windows (parallelized across windows)
    total_windows = len(signal) - window_samples + 1
    logging.info(f"Fitting AR({ar_order}) models across {total_windows} windows using n_jobs={n_jobs}...")

    windows = np.lib.stride_tricks.sliding_window_view(signal, window_samples)

    t0 = time.time()
    with tqdm_joblib(tqdm(total=total_windows, desc="AR fitting", unit="win")):
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_window)(windows[start], ar_order, target_fs, spindle_band)
            for start in range(total_windows)
        )
    logging.info(f"AR fitting complete in {time.time() - t0:.2f}s "
                 f"({total_windows / (time.time() - t0):.1f} windows/sec).")

    # Preallocate and fill from results
    r_timeseries = np.empty(total_windows, dtype=float)
    f_timeseries = np.empty(total_windows, dtype=float)
    for i, (r_val, f_val) in enumerate(results):
        r_timeseries[i] = r_val
        f_timeseries[i] = f_val

    # 3. Detect events using upper/lower thresholds
    logging.info("Scanning r-timeseries for spindle events "
                 f"(upper={upper_threshold}, lower={lower_threshold})...")
    t0 = time.time()
    events = []
    in_event = False
    start_idx = None

    for i, r in enumerate(r_timeseries):

        # Start event
        if not in_event and r >= upper_threshold:
            in_event = True
            start_idx = i

        # Continue event while r >= lower threshold
        elif in_event and r < lower_threshold:

            end_idx = i - 1

            # Find maximum r within event
            event_slice = slice(start_idx, end_idx + 1)
            peak_relative = np.argmax(r_timeseries[event_slice])
            peak_idx = start_idx + peak_relative

            max_r = r_timeseries[peak_idx]
            peak_frequency = f_timeseries[peak_idx]

            # Convert window indices to seconds
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

    # 4. Handle event extending to end of recording
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
    
    # Calculate stats for all valid r-values across the entire recording
    valid_r = r_timeseries[~np.isnan(f_timeseries)]
    if valid_r.size > 0:
        logging.info(f"--- Global R Statistics (in spindle band) ---")
        logging.info(f"Min: {valid_r.min():.3f} | Max: {valid_r.max():.3f} | Avg: {valid_r.mean():.3f} | Std: {valid_r.std():.3f}")
    
    logging.info(f"Total detection time: {time.time() - t_start:.2f}s.")

    return np.asarray(events)
