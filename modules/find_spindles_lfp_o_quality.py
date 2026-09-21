import contextlib
import logging
import time
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm

from modules.ephys_preprocessing import bandpass_filter, downsampling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================================
# joblib + tqdm
# ============================================================================

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that makes joblib progress update tqdm.
    """
    class TqdmBatchCompletionCallback(
        joblib.parallel.BatchCompletionCallBack
    ):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = (
        TqdmBatchCompletionCallback
    )

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ============================================================================
# AR fitting
# ============================================================================

def _fit_window_all_poles(
    window,
    ar_order,
    target_fs,
):
    """
    Fit an AR model to one window and return all positive-frequency poles.

    Returns
    -------
    frequencies : np.ndarray
        Positive-frequency pole frequencies in Hz.

    radii : np.ndarray
        Pole radii.

    Notes
    -----
    The AR model is:

        x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

    and the characteristic polynomial is:

        z^p - a1*z^(p-1) - ... - ap = 0

    For each complex pole:

        z = r * exp(i*phi)

    frequency is:

        f = phi * fs / (2*pi)

    We keep only positive-frequency poles, i.e. one member of
    each conjugate pair.
    """
    try:
        window = np.asarray(window, dtype=float)

        if window.ndim != 1:
            raise ValueError("AR window must be one-dimensional.")

        if not np.all(np.isfinite(window)):
            return np.empty(0), np.empty(0)

        # Keep demean=False to match the original implementation and
        # the stated AR formulation.
        a, _ = sm.regression.linear_model.burg(
            window,
            order=ar_order,
            demean=False,
        )

        a = np.asarray(a, dtype=float)

        # AR polynomial:
        #
        # z^p - a1*z^(p-1) - ... - ap
        #
        # np.r_ produces:
        #
        # [1, -a1, -a2, ..., -ap]
        poles = np.roots(
            np.r_[1.0, -a]
        )

        # Keep one pole from each complex conjugate pair.
        positive_frequency = np.imag(poles) > 0

        poles = poles[positive_frequency]

        if len(poles) == 0:
            return np.empty(0), np.empty(0)

        frequencies = (
            np.angle(poles)
            * target_fs
            / (2.0 * np.pi)
        )

        radii = np.abs(poles)

        # Sort by frequency so that output is deterministic.
        order = np.argsort(frequencies)

        frequencies = frequencies[order]
        radii = radii[order]

        return (
            frequencies.astype(float),
            radii.astype(float),
        )

    except Exception:
        return np.empty(0), np.empty(0)


def _fit_window(
    window,
    ar_order,
    target_fs,
    spindle_band,
):
    """
    Backward-compatible single-window interface.

    Returns the strongest spindle-band pole:

        (best_r, best_freq)

    This function is intentionally retained because it may be used
    elsewhere in the application.

    The detector uses the strongest spindle-band pole in each window.
    """
    frequencies, radii = _fit_window_all_poles(
        window,
        ar_order,
        target_fs,
    )

    if len(radii) == 0:
        return 0.0, np.nan

    mask = (
        (frequencies >= spindle_band[0])
        & (frequencies <= spindle_band[1])
    )

    if not np.any(mask):
        return 0.0, np.nan

    candidate_frequencies = frequencies[mask]
    candidate_radii = radii[mask]

    best = np.argmax(candidate_radii)

    return (
        float(candidate_radii[best]),
        float(candidate_frequencies[best]),
    )


# ============================================================================
# Original-compatible window fitting
# ============================================================================

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
    """
    Fit AR windows and return the strongest spindle-band pole per window.

    This preserves the original interface.
    """
    if len(starts) == 0:
        return np.array([]), np.array([])

    def fit_one(s):
        return _fit_window(
            signal[
                s : s + window_samples
            ],
            ar_order,
            target_fs,
            spindle_band,
        )

    if verbose:
        with tqdm_joblib(
            tqdm(
                total=len(starts),
                desc=desc,
                unit="win",
            )
        ):
            results = Parallel(
                n_jobs=n_jobs,
                prefer="processes",
            )(
                delayed(fit_one)(s)
                for s in starts
            )
    else:
        results = Parallel(
            n_jobs=n_jobs,
            prefer="processes",
        )(
            delayed(fit_one)(s)
            for s in starts
        )

    r_vals = np.array(
        [r for r, f in results],
        dtype=float,
    )

    f_vals = np.array(
        [f for r, f in results],
        dtype=float,
    )

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
    """
    Backward-compatible public function.

    Fits AR models over overlapping windows and returns:

        r_timeseries
        f_timeseries
        starts

    IMPORTANT:
    This public function retains the historical behavior of returning
    the strongest spindle-band pole for each window.

    The main spindle detector uses the all-pole representation internally.
    """
    window_samples = int(
        window_sec * target_fs
    )

    if len(signal) < window_samples:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    total_windows = (
        len(signal)
        - window_samples
        + 1
    )

    starts = np.arange(
        0,
        total_windows,
        stride_samples,
    )

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

    return (
        r_vals,
        f_vals,
        starts,
    )


# ============================================================================
# All-pole fitting used by the corrected detector
# ============================================================================

def _fit_all_poles_at_start(
    signal,
    start,
    window_samples,
    ar_order,
    target_fs,
):
    """
    Fit one AR window and return all positive-frequency poles.
    """
    return _fit_window_all_poles(
        signal[
            start : start + window_samples
        ],
        ar_order,
        target_fs,
    )


def _fit_all_pole_windows(
    signal,
    starts,
    window_samples,
    ar_order,
    target_fs,
    n_jobs,
    verbose,
):
    """
    Fit all AR poles for all windows.

    Returns
    -------
    frequencies : list[np.ndarray]
        frequencies[i] contains the positive-frequency poles
        in window i.

    radii : list[np.ndarray]
        radii[i] contains the corresponding radii.
    """
    if len(starts) == 0:
        return [], []

    def fit_one(start):
        return _fit_all_poles_at_start(
            signal,
            start,
            window_samples,
            ar_order,
            target_fs,
        )

    if verbose:
        with tqdm_joblib(
            tqdm(
                total=len(starts),
                desc="AR fitting",
                unit="win",
            )
        ):
            results = Parallel(
                n_jobs=n_jobs,
                prefer="processes",
            )(
                delayed(fit_one)(start)
                for start in starts
            )
    else:
        results = Parallel(
            n_jobs=n_jobs,
            prefer="processes",
        )(
            delayed(fit_one)(start)
            for start in starts
        )

    frequencies = [
        result[0]
        for result in results
    ]

    radii = [
        result[1]
        for result in results
    ]

    return frequencies, radii


# ============================================================================
# Paper-faithful per-window pole selection
# ============================================================================

def _select_strongest_spindle_pole(
    frequencies,
    radii,
    spindle_band,
):
    """
    Select the strongest spindle-band pole in ONE AR window.

    A real AR(8) model has eight roots. Complex roots occur as
    conjugate pairs, so there can be up to four oscillatory modes.
    For spindle detection/characterization, we keep the positive-frequency
    member of each pair, restrict to the spindle band, and select the pole
    with the largest radius R.

    No pole identity is tracked between windows.

    Returns
    -------
    r : float
        Largest spindle-band pole radius for this window.
    f : float
        Frequency (Hz) of that same pole.
    """
    if len(frequencies) == 0:
        return 0.0, np.nan

    low_f, high_f = spindle_band

    mask = (
        (frequencies >= low_f)
        & (frequencies <= high_f)
        & np.isfinite(frequencies)
        & np.isfinite(radii)
    )

    if not np.any(mask):
        return 0.0, np.nan

    candidate_frequencies = frequencies[mask]
    candidate_radii = radii[mask]

    best = int(np.argmax(candidate_radii))

    return (
        float(candidate_radii[best]),
        float(candidate_frequencies[best]),
    )


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
    """
    Fit one AR model per window.

    For each timestamp/window, retain ONLY the strongest pole in the
    spindle band. The frequency returned is the frequency of that
    strongest-R pole in the same window.

    There is deliberately NO frequency-continuity or pole-identity
    tracking between adjacent windows.
    """
    if len(starts) == 0:
        return np.array([]), np.array([])

    def fit_one(s):
        window = signal[s : s + window_samples]

        frequencies, radii = _fit_window_all_poles(
            window,
            ar_order,
            target_fs,
        )

        return _select_strongest_spindle_pole(
            frequencies,
            radii,
            spindle_band,
        )

    if verbose:
        with tqdm_joblib(
            tqdm(
                total=len(starts),
                desc=desc,
                unit="win",
            )
        ):
            results = Parallel(
                n_jobs=n_jobs,
                prefer="processes",
            )(
                delayed(fit_one)(s)
                for s in starts
            )
    else:
        results = Parallel(
            n_jobs=n_jobs,
            prefer="processes",
        )(
            delayed(fit_one)(s)
            for s in starts
        )

    r_vals = np.array(
        [r for r, f in results],
        dtype=float,
    )

    f_vals = np.array(
        [f for r, f in results],
        dtype=float,
    )

    return r_vals, f_vals


# ============================================================================
# Public AR analysis
# ============================================================================

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
    """
    Fit AR models over overlapping windows.

    Returns
    -------
    r_timeseries : np.ndarray
        Strongest spindle-band pole radius in each window.

    f_timeseries : np.ndarray
        Frequency (Hz) of that same strongest-R pole in each window.

    starts : np.ndarray
        Starting sample of each AR window.

    Notes
    -----
    Each window is evaluated independently.

    The frequency is allowed to change freely from one window to the next.
    For example, one window can select 11 Hz and the next can select 12 Hz.

    This function does NOT track pole identities across time.
    """
    window_samples = int(
        round(window_sec * target_fs)
    )

    if len(signal) < window_samples:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    total_windows = (
        len(signal)
        - window_samples
        + 1
    )

    starts = np.arange(
        0,
        total_windows,
        stride_samples,
    )

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

    return (
        r_vals,
        f_vals,
        starts,
    )


# ============================================================================
# Event extraction
# ============================================================================

def _finalize_event(
    events,
    r,
    f,
    sample_starts,
    target_fs,
    window_samples,
    start_i,
    end_i,
):
    """
    Finalize one event from a single pole trajectory.

    Output format is deliberately unchanged:

        [start_time,
         peak_time,
         end_time,
         duration,
         max_r,
         peak_frequency]
    """
    if end_i < start_i:
        return

    r_segment = np.asarray(
        r[start_i : end_i + 1],
        dtype=float,
    )

    if len(r_segment) == 0:
        return

    finite_mask = np.isfinite(r_segment)

    if not np.any(finite_mask):
        return

    # Replace invalid values by -inf only for finding the maximum.
    values_for_argmax = r_segment.copy()
    values_for_argmax[~finite_mask] = -np.inf

    peak_relative = int(
        np.argmax(values_for_argmax)
    )

    peak_i = start_i + peak_relative

    max_r = float(r[peak_i])
    peak_frequency = float(f[peak_i])

    # Timestamp each AR estimate at the center of its window.
    half_win = window_samples / 2.0

    start_time = (
        sample_starts[start_i]
        + half_win
    ) / target_fs

    end_time = (
        sample_starts[end_i]
        + half_win
    ) / target_fs

    peak_time = (
        sample_starts[peak_i]
        + half_win
    ) / target_fs

    duration = end_time - start_time

    events.append(
        [
            start_time,
            peak_time,
            end_time,
            duration,
            max_r,
            peak_frequency,
        ]
    )


def _detect_events_in_region(
    r,
    f,
    sample_starts,
    target_fs,
    window_samples,
    upper_threshold,
    lower_threshold,
):
    """
    Detect events using upper/lower hysteresis.

    Event starts when:

        r > upper_threshold

    Once active, the event continues while:

        r >= lower_threshold

    A value below lower_threshold ends the event.

    This preserves the original function interface.
    """
    events = []

    n = len(r)
    i = 0

    while i < n:

        # Ignore invalid values.
        if (
            not np.isfinite(r[i])
            or not np.isfinite(f[i])
        ):
            i += 1
            continue

        # --------------------------------------------------------
        # Event onset.
        # --------------------------------------------------------
        if r[i] > upper_threshold:

            start_i = i
            end_i = i

            # ----------------------------------------------------
            # Continue through the event.
            # ----------------------------------------------------
            while end_i < n - 1:

                next_i = end_i + 1

                if (
                    not np.isfinite(r[next_i])
                    or not np.isfinite(f[next_i])
                ):
                    break

                # Strong continuation.
                if r[next_i] > upper_threshold:
                    end_i = next_i
                    continue

                # Hysteresis region.
                if r[next_i] >= lower_threshold:
                    end_i = next_i
                    continue

                # Below lower threshold:
                # event terminates.
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


# ============================================================================
# Event detection
# ============================================================================

def _detect_events_in_region(
    r,
    f,
    sample_starts,
    target_fs,
    window_samples,
    upper_threshold,
    lower_threshold,
):
    """
    Detect events from the strongest spindle-band R at each window.

    Event starts when:
        R > upper_threshold

    Once active, it continues while:
        R >= lower_threshold

    The frequency f is simply the frequency associated with the
    strongest-R pole in each individual window. It is NOT used to
    track a pole across windows.
    """
    events = []

    n = len(r)
    i = 0

    while i < n:

        if (
            not np.isfinite(r[i])
            or not np.isfinite(f[i])
        ):
            i += 1
            continue

        if r[i] > upper_threshold:

            start_i = i
            end_i = i

            while end_i < n - 1:

                next_i = end_i + 1

                if (
                    not np.isfinite(r[next_i])
                    or not np.isfinite(f[next_i])
                ):
                    break

                if r[next_i] >= lower_threshold:
                    end_i = next_i
                    continue

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


# ============================================================================
# Duration filtering
# ============================================================================

def filter_events_by_duration(
    events_list,
    min_duration_sec=0.4,
    max_duration_sec=3.5,
):
    """
    Keep events within the requested duration range.

    Set either limit to None to disable that bound.

    Examples
    --------
    No minimum:

        min_duration_sec=None

    No maximum:

        max_duration_sec=None

    No duration filtering:

        min_duration_sec=None,
        max_duration_sec=None
    """
    if not events_list:
        return np.empty(
            (0, 6),
            dtype=float,
        )

    events = np.asarray(
        events_list,
        dtype=float,
    )

    durations = events[:, 3]

    mask = np.ones(
        len(events),
        dtype=bool,
    )

    if min_duration_sec is not None:
        mask &= (
            durations >= min_duration_sec
        )

    if max_duration_sec is not None:
        mask &= (
            durations <= max_duration_sec
        )

    return events[mask]


# ============================================================================
# o-Quality helper
# ============================================================================

def classify_o_quality(
    max_r,
    upper_threshold=0.92,
):
    """
    Classify an event according to the paper's o-Quality bins.

    The paper's bins are defined relative to the 0.92 detection
    threshold:

        0.92 <= r < 0.93 -> oQ1
        0.93 <= r < 0.94 -> oQ2
        0.94 <= r < 0.95 -> oQ3
        r >= 0.95        -> oQ4

    If a custom detection threshold is used, values below that
    threshold are returned as None.

    Returns
    -------
    str or None
    """
    if not np.isfinite(max_r):
        return None

    if max_r < upper_threshold:
        return None

    # The paper's oQ boundaries are tied to the standard 0.92
    # threshold. If a different threshold is supplied, retain
    # the fixed paper bins rather than silently redefining oQ.
    if max_r < 0.93:
        return "oQ1"

    if max_r < 0.94:
        return "oQ2"

    if max_r < 0.95:
        return "oQ3"

    return "oQ4"


# ============================================================================
# Main detector
# ============================================================================

def find_spindles_lfp(
    raw_signal,
    fs,
    target_fs=128,
    ar_order=8,
    window_sec=1.0,
    stride_samples=1,
    upper_threshold=0.92,
    lower_threshold=0.90,
    spindle_band=(10, 15),
    min_duration_sec=None,
    max_duration_sec=None,
    n_jobs=1,
    lowcut=0.1,
    highcut=100.0,
    verbose=False,
):
    """
    Detect spindle-like events using the paper's AR(8) approach.

    Core method
    -----------
    1. Band-pass the signal (0.1-100 Hz by default).
    2. Resample to 128 Hz.
    3. Fit an AR(8) model in each overlapping 1-s window.
    4. Calculate the AR roots/poles.
    5. Keep positive-frequency poles in the 10-15 Hz spindle band.
    6. Select the pole with the largest radius R IN THAT WINDOW.
    7. Use that window's R for hysteresis detection.
    8. The reported peak frequency is the frequency of the window's
       maximum-R pole.

    There is NO frequency-continuity tracking. If the strongest pole is
    11 Hz in one window and 12 Hz in the next, the returned frequency
    simply changes from 11 to 12 Hz.

    Parameters
    ----------
    upper_threshold : float, default=0.92
        R threshold for event onset.

    lower_threshold : float, default=0.90
        R threshold for continuation.

    min_duration_sec, max_duration_sec : float or None
        Optional duration filters. The paper does not specify these as
        part of the AR threshold detector, so they default to None.

    Returns
    -------
    np.ndarray
        Shape (N, 6):

            0 = start_time
            1 = peak_time
            2 = end_time
            3 = duration
            4 = max_r
            5 = peak_frequency
    """
    t_start = time.time()

    raw_signal = np.asarray(
        raw_signal,
        dtype=float,
    )

    if raw_signal.ndim != 1:
        raise ValueError(
            "raw_signal must be a one-dimensional array."
        )

    if not np.all(np.isfinite(raw_signal)):
        raise ValueError(
            "raw_signal contains NaN or infinite values."
        )

    if fs <= 0:
        raise ValueError("fs must be positive.")

    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    if ar_order < 1:
        raise ValueError("ar_order must be >= 1.")

    if window_sec <= 0:
        raise ValueError("window_sec must be > 0.")

    if stride_samples < 1:
        raise ValueError(
            "stride_samples must be >= 1."
        )

    if len(spindle_band) != 2:
        raise ValueError(
            "spindle_band must be a (low, high) tuple."
        )

    spindle_low, spindle_high = spindle_band

    if spindle_low >= spindle_high:
        raise ValueError(
            "spindle_band lower bound must be less than upper bound."
        )

    if lower_threshold > upper_threshold:
        raise ValueError(
            "lower_threshold must be <= upper_threshold."
        )

    if lowcut <= 0:
        raise ValueError("lowcut must be > 0.")

    if highcut <= lowcut:
        raise ValueError(
            "highcut must be greater than lowcut."
        )

    if highcut >= fs / 2:
        raise ValueError(
            f"highcut ({highcut}) must be below the original "
            f"Nyquist frequency ({fs / 2})."
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    filtered_signal = bandpass_filter(
        raw_signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
    )

    signal = downsampling(
        filtered_signal,
        fs,
        target_fs,
    )

    window_samples = int(
        round(window_sec * target_fs)
    )

    if window_samples <= ar_order:
        raise ValueError(
            "AR window is too short for the requested AR order."
        )

    if len(signal) < window_samples:
        return np.empty(
            (0, 6),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # One independently evaluated AR window at every requested start.
    # ------------------------------------------------------------------

    total_windows = (
        len(signal)
        - window_samples
        + 1
    )

    starts = np.arange(
        0,
        total_windows,
        stride_samples,
    )

    r_values, f_values = _fit_windows_at_starts(
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

    if len(r_values) == 0:
        return np.empty(
            (0, 6),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Hysteresis detection on the strongest-R-per-window series.
    # ------------------------------------------------------------------

    raw_events = _detect_events_in_region(
        r_values,
        f_values,
        starts,
        target_fs,
        window_samples,
        upper_threshold,
        lower_threshold,
    )

    # ------------------------------------------------------------------
    # Optional duration filtering.
    # ------------------------------------------------------------------

    final_events = filter_events_by_duration(
        raw_events,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )

    logging.info(
        "Raw candidate events: %d -> Filtered events: %d (%.2fs)",
        len(raw_events),
        len(final_events),
        time.time() - t_start,
    )

    return final_events

