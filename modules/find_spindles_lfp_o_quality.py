import contextlib
import logging
import time

import joblib
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm

from modules.ephys_preprocessing import bandpass_filter, downsampling

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def _fit_windows_at_starts(
    signal,
    starts,
    window_samples,
    ar_order,
    target_fs,
    spindle_band,
    n_jobs,
    verbose,
    desc,
):
    """Fit AR windows starting at arbitrary sample indices."""
    if len(starts) == 0:
        return np.array([]), np.array([])

    if verbose:
        with tqdm_joblib(tqdm(total=len(starts), desc=desc, unit="win")):
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_fit_window)(
                    signal[s : s + window_samples], ar_order, target_fs, spindle_band
                )
                for s in starts
            )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_window)(
                signal[s : s + window_samples], ar_order, target_fs, spindle_band
            )
            for s in starts
        )

    r_vals = np.array([r for r, f in results])
    f_vals = np.array([f for r, f in results])
    return r_vals, f_vals


def fit_ar_on_prepared_signal(
    signal,
    target_fs=128,
    ar_order=8,
    window_sec=1.0,
    stride_samples=1,
    spindle_band=(10, 15),
    n_jobs=1,
    verbose=False,
):
    window_samples = int(window_sec * target_fs)
    if len(signal) < window_samples:
        return np.array([]), np.array([]), np.array([])

    total_windows = len(signal) - window_samples + 1
    starts = np.arange(0, total_windows, stride_samples)
    r_vals, f_vals = _fit_windows_at_starts(
        signal,
        starts,
        window_samples,
        ar_order,
        target_fs,
        spindle_band,
        n_jobs,
        verbose,
        desc="AR fitting",
    )
    return r_vals, f_vals, starts


def _detect_events_in_region(
    r, f, sample_starts, target_fs, window_samples,
    upper_threshold, lower_threshold
):
    """
    Detect oscillatory events using upper/lower hysteresis.

    upper_threshold (rb):
        Defines event onset and offset.

    lower_threshold (ra):
        Determines whether consecutive detections belong to the
        same continuous event or should be split.
    """
    events = []
    n = len(r)
    i = 0

    while i < n:

        # Event starts when r exceeds the upper threshold rb
        if r[i] > upper_threshold:

            start_i = i
            end_i = i

            while end_i < n - 1:

                next_i = end_i + 1

                # Clearly still inside the event
                if r[next_i] > upper_threshold:
                    end_i = next_i

                # Between lower and upper threshold:
                # keep looking because this may be a transient fluctuation
                elif r[next_i] >= lower_threshold:
                    end_i = next_i

                # Below lower threshold -> split/end event
                else:
                    break

            _finalize_event(
                events,
                r,
                f,
                sample_starts,
                target_fs,
                window_samples,
                start_i,
                end_i,
            )

            i = end_i + 1

        else:
            i += 1

    return events

def _finalize_event(
    events, r, f, sample_starts, target_fs, window_samples, start_i, end_i
):
    peak_relative = np.argmax(r[start_i : end_i + 1])
    peak_i = start_i + peak_relative

    max_r = r[peak_i]
    peak_frequency = f[peak_i]

    half_win = window_samples / 2.0
    start_time = (sample_starts[start_i] + half_win) / target_fs
    end_time = (sample_starts[end_i] + half_win) / target_fs
    peak_time = (sample_starts[peak_i] + half_win) / target_fs
    duration = end_time - start_time

    events.append([start_time, peak_time, end_time, duration, max_r, peak_frequency])


def filter_events_by_duration(
    events_list,
    min_duration_sec=0.4,
    max_duration_sec=3.5,
):
    """Keep events within the accepted spindle duration range."""
    if not events_list:
        return np.empty((0, 6))

    events = np.asarray(events_list)

    durations = events[:, 3]

    mask = (
        (durations >= min_duration_sec)
        & (durations <= max_duration_sec)
    )

    return events[mask]

def find_spindles_lfp(
    raw_signal, fs, target_fs=128, ar_order=8, window_sec=1.0,
    stride_samples=2, upper_threshold=0.80, lower_threshold=0.65,
    spindle_band=(9, 20), min_duration_sec=0.4,
    max_duration_sec=3.5, n_jobs=1,
):
    t_start = time.time()
    filtered_signal = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=fs)
    signal = downsampling(filtered_signal, fs, target_fs)
    window_samples = int(window_sec * target_fs)

    r_timeseries, f_timeseries, starts = fit_ar_on_prepared_signal(
        signal,
        target_fs=target_fs,
        ar_order=ar_order,
        window_sec=window_sec,
        stride_samples=stride_samples,
        spindle_band=spindle_band,
        n_jobs=n_jobs,
        verbose=False,
    )
    if len(r_timeseries) == 0:
        return np.empty((0, 6))

    raw_events = _detect_events_in_region(
        r_timeseries, f_timeseries, starts,
        target_fs, window_samples, upper_threshold, lower_threshold,
    )

    final_events = filter_events_by_duration(
        raw_events,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )

    # Re-enable the merge tracking log
    logging.info(
        f"Raw candidate events: {len(raw_events)} -> "
        f"Filtered events: {len(final_events)} "
        f"({time.time() - t_start:.2f}s)"
    )

    return final_events
