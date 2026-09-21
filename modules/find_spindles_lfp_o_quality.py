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

    The main detector below uses _fit_window_all_poles() so that
    individual oscillatory modes are not lost.
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
# Pole tracking
# ============================================================================

def _track_poles_by_frequency(
    frequencies_per_window,
    radii_per_window,
    spindle_band,
    max_frequency_jump_hz=3.0,
):
    """
    Track oscillatory poles through time using frequency continuity.

    Parameters
    ----------
    frequencies_per_window:
        List of arrays containing positive-frequency poles.

    radii_per_window:
        Corresponding pole radii.

    spindle_band:
        Only poles in this frequency range are tracked.

    max_frequency_jump_hz:
        Maximum frequency difference allowed between consecutive
        observations belonging to the same trajectory.

    Returns
    -------
    tracks : list of dict
        Each dictionary contains:

            "frequency"
            "radius"
            "window_index"

    Notes
    -----
    The paper does not specify an exact pole identity-tracking
    algorithm. This frequency-continuity tracker is therefore an
    implementation choice.

    It is preferable to taking the maximum-r pole independently
    at every window because the latter can switch between different
    oscillatory modes.
    """
    active_tracks = []
    finished_tracks = []

    for window_idx, (
        frequencies,
        radii,
    ) in enumerate(
        zip(
            frequencies_per_window,
            radii_per_window,
        )
    ):

        if len(frequencies) == 0:
            # End all active tracks when there is no usable pole.
            finished_tracks.extend(active_tracks)
            active_tracks = []
            continue

        # Keep only poles in requested spindle band.
        mask = (
            (frequencies >= spindle_band[0])
            & (frequencies <= spindle_band[1])
            & np.isfinite(frequencies)
            & np.isfinite(radii)
        )

        current_frequencies = frequencies[mask]
        current_radii = radii[mask]

        if len(current_frequencies) == 0:
            finished_tracks.extend(active_tracks)
            active_tracks = []
            continue

        # Each current pole can be assigned to at most one track.
        used_current = set()

        assignments = []

        # Match existing tracks to current poles using nearest
        # frequency difference.
        candidate_pairs = []

        for track_idx, track in enumerate(active_tracks):

            previous_frequency = track["frequency"][-1]

            for current_idx, current_frequency in enumerate(
                current_frequencies
            ):
                difference = abs(
                    current_frequency
                    - previous_frequency
                )

                if difference <= max_frequency_jump_hz:
                    candidate_pairs.append(
                        (
                            difference,
                            track_idx,
                            current_idx,
                        )
                    )

        # Nearest-frequency assignments first.
        candidate_pairs.sort(
            key=lambda x: x[0]
        )

        assigned_tracks = set()

        for (
            difference,
            track_idx,
            current_idx,
        ) in candidate_pairs:

            if track_idx in assigned_tracks:
                continue

            if current_idx in used_current:
                continue

            assignments.append(
                (
                    track_idx,
                    current_idx,
                )
            )

            assigned_tracks.add(track_idx)
            used_current.add(current_idx)

        # Update matched tracks.
        for (
            track_idx,
            current_idx,
        ) in assignments:

            track = active_tracks[track_idx]

            track["frequency"].append(
                float(
                    current_frequencies[current_idx]
                )
            )

            track["radius"].append(
                float(
                    current_radii[current_idx]
                )
            )

            track["window_index"].append(
                window_idx
            )

        # Tracks that were not matched are finished.
        unmatched_tracks = []

        for track_idx, track in enumerate(
            active_tracks
        ):
            if track_idx not in assigned_tracks:
                unmatched_tracks.append(track)

        finished_tracks.extend(
            unmatched_tracks
        )

        active_tracks = [
            track
            for track_idx, track in enumerate(
                active_tracks
            )
            if track_idx in assigned_tracks
        ]

        # Create new tracks for unmatched current poles.
        for current_idx in range(
            len(current_frequencies)
        ):
            if current_idx in used_current:
                continue

            active_tracks.append(
                {
                    "frequency": [
                        float(
                            current_frequencies[
                                current_idx
                            ]
                        )
                    ],
                    "radius": [
                        float(
                            current_radii[
                                current_idx
                            ]
                        )
                    ],
                    "window_index": [
                        window_idx
                    ],
                }
            )

    finished_tracks.extend(active_tracks)

    return finished_tracks


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
# Corrected multi-pole event detection
# ============================================================================

def _detect_events_from_tracks(
    tracks,
    starts,
    target_fs,
    window_samples,
    upper_threshold,
    lower_threshold,
):
    """
    Detect events separately on every tracked AR oscillator.

    This is the important correction compared with the old code.

    The old implementation first selected:

        max(r) across all spindle-band poles

    for every window.

    That can cause the time series to jump from one oscillator
    to another. Here, each pole trajectory is treated separately.
    """
    all_events = []

    for track_id, track in enumerate(tracks):

        window_indices = np.asarray(
            track["window_index"],
            dtype=int,
        )

        frequencies = np.asarray(
            track["frequency"],
            dtype=float,
        )

        radii = np.asarray(
            track["radius"],
            dtype=float,
        )

        if len(window_indices) == 0:
            continue

        # The generic event detector expects a contiguous local
        # series. Check for gaps.
        #
        # If there is a gap, split the track into contiguous pieces.
        split_points = np.where(
            np.diff(window_indices) > 1
        )[0]

        segments = np.split(
            np.arange(len(window_indices)),
            split_points + 1,
        )

        for segment in segments:

            if len(segment) == 0:
                continue

            local_f = frequencies[segment]
            local_r = radii[segment]
            local_starts = starts[
                window_indices[segment]
            ]

            local_events = _detect_events_in_region(
                local_r,
                local_f,
                local_starts,
                target_fs,
                window_samples,
                upper_threshold,
                lower_threshold,
            )

            for event in local_events:
                # Keep original six-column event interface.
                all_events.append(event)

    # Sort by event onset.
    if not all_events:
        return []

    all_events.sort(
        key=lambda event: event[0]
    )

    return all_events


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
    min_duration_sec=0.4,
    max_duration_sec=3.5,
    n_jobs=1,
    lowcut=0.1,
    highcut=100.0,
    verbose=False,
    max_frequency_jump_hz=3.0,
):
    """
    Detect spindle-like oscillatory events using an AR model.

    Parameters
    ----------
    raw_signal : array-like
        Raw LFP/EEG signal.

    fs : float
        Original sampling frequency.

    target_fs : float, default=128
        Sampling rate used for AR analysis.

    ar_order : int, default=8
        AR model order.

    window_sec : float, default=1.0
        AR window duration.

    stride_samples : int, default=1
        Number of target_fs samples between successive windows.

        For the paper's original implementation:
            stride_samples=1

        This remains configurable.

    upper_threshold : float, default=0.92
        Event onset threshold.

        This remains configurable because your recording
        devices/regions may require different thresholds.

    lower_threshold : float, default=0.90
        Event continuation/splitting threshold.

        Must normally be <= upper_threshold.

    spindle_band : tuple, default=(10, 15)
        Frequency range used for spindle detection.

        This remains configurable.

    min_duration_sec : float or None, default=0.4
        Minimum event duration.

        Set None to disable the lower duration bound.

    max_duration_sec : float or None, default=3.5
        Maximum event duration.

        Set None to disable the upper duration bound.

    n_jobs : int, default=1
        Number of parallel jobs.

    lowcut : float, default=0.1
        High-pass cutoff.

        This is now configurable without breaking existing calls.

    highcut : float, default=100.0
        Low-pass cutoff.

        This is now configurable without breaking existing calls.

    verbose : bool, default=False
        Show AR-fitting progress.

    max_frequency_jump_hz : float, default=3.0
        Maximum frequency change between consecutive windows
        when tracking an AR oscillatory pole.

        This is an implementation choice because the paper does
        not specify a pole identity-tracking threshold.

    Returns
    -------
    final_events : np.ndarray
        Shape (N, 6):

            column 0 = start_time
            column 1 = peak_time
            column 2 = end_time
            column 3 = duration
            column 4 = max_r
            column 5 = peak_frequency

    Notes
    -----
    The public return format is deliberately unchanged from the
    previous implementation.

    Internally, however, all spindle-band AR poles are retained and
    tracked separately before event detection.
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
        raise ValueError(
            "fs must be positive."
        )

    if target_fs <= 0:
        raise ValueError(
            "target_fs must be positive."
        )

    if ar_order < 1:
        raise ValueError(
            "ar_order must be >= 1."
        )

    if window_sec <= 0:
        raise ValueError(
            "window_sec must be > 0."
        )

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
        raise ValueError(
            "lowcut must be > 0."
        )

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
        round(
            window_sec * target_fs
        )
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
    # Window locations
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

    # ------------------------------------------------------------------
    # Fit all AR poles
    # ------------------------------------------------------------------

    frequencies_per_window, radii_per_window = (
        _fit_all_pole_windows(
            signal,
            starts,
            window_samples,
            ar_order,
            target_fs,
            n_jobs,
            verbose,
        )
    )

    if len(frequencies_per_window) == 0:
        return np.empty(
            (0, 6),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Track individual spindle-frequency oscillators
    # ------------------------------------------------------------------

    tracks = _track_poles_by_frequency(
        frequencies_per_window,
        radii_per_window,
        spindle_band=spindle_band,
        max_frequency_jump_hz=max_frequency_jump_hz,
    )

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    raw_events = _detect_events_from_tracks(
        tracks,
        starts,
        target_fs,
        window_samples,
        upper_threshold,
        lower_threshold,
    )

    # ------------------------------------------------------------------
    # Optional duration filtering
    # ------------------------------------------------------------------

    final_events = filter_events_by_duration(
        raw_events,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )

    logging.info(
        "Raw candidate events: %d -> "
        "Filtered events: %d "
        "(%.2fs)",
        len(raw_events),
        len(final_events),
        time.time() - t_start,
    )

    return final_events
