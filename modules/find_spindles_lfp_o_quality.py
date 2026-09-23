"""
Paper-faithful AR spindle detection based on:

1) Olbrich & Achermann (2005), Journal of Sleep Research
2) Blanco-Duque et al. (2024), Science Advances

Important implementation notes
------------------------------
- AR(8), Burg estimation, 1-s windows.
- Windows are shifted by one sample by default.
- The detector evaluates ALL positive-frequency AR poles in the spindle band;
  it does not collapse each window to the single strongest pole.
- No spindle-band filtering is used before the AR model.
- Upper/lower hysteresis is used: rb starts an event; ra keeps a detection
  continuous/merges nearby detections.
- The public interface of find_spindles_lfp() is unchanged.
- fit_ar_on_prepared_signal() is retained for backward compatibility and
  continues to return the strongest spindle-band pole per window, because
  changing that public helper would break existing application code.
- Pole matching across adjacent windows is needed operationally because the
  papers refer to continuity "within a specific frequency", but they do not
  specify an exact matching algorithm. The implementation below uses
  nearest-frequency one-to-one matching with an internal 2-Hz continuity
  limit. This is an implementation detail, not a value claimed by either
  paper.
"""

import contextlib
import logging
import time
from typing import Optional

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

def _fit_window_all_poles(window, ar_order, target_fs):
    """
    Fit one Burg AR model and return all positive-frequency poles.

    Returns
    -------
    frequencies : np.ndarray
        Positive-frequency pole frequencies in Hz.

    radii : np.ndarray
        Corresponding pole radii R.
    """
    try:
        window = np.asarray(window, dtype=float)

        if window.ndim != 1:
            raise ValueError("AR window must be one-dimensional.")

        if not np.all(np.isfinite(window)):
            return np.empty(0), np.empty(0)

        a, _ = sm.regression.linear_model.burg(
            window,
            order=ar_order,
            demean=False,
        )

        a = np.asarray(a, dtype=float)

        # AR polynomial:
        # z^p - a1*z^(p-1) - ... - ap = 0
        poles = np.roots(np.r_[1.0, -a])

        # One member of each complex-conjugate pair.
        poles = poles[np.imag(poles) > 0]

        if len(poles) == 0:
            return np.empty(0), np.empty(0)

        frequencies = (
            np.angle(poles) * target_fs / (2.0 * np.pi)
        )
        radii = np.abs(poles)

        order = np.argsort(frequencies)

        return (
            frequencies[order].astype(float),
            radii[order].astype(float),
        )

    except Exception:
        return np.empty(0), np.empty(0)


def _fit_window(window, ar_order, target_fs, spindle_band):
    """
    Backward-compatible single-window helper.

    Returns the strongest spindle-band pole:
        (R, frequency)
    """
    frequencies, radii = _fit_window_all_poles(
        window,
        ar_order,
        target_fs,
    )

    if len(radii) == 0:
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

    candidate_f = frequencies[mask]
    candidate_r = radii[mask]

    best = int(np.argmax(candidate_r))

    return float(candidate_r[best]), float(candidate_f[best])


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
    Backward-compatible helper.

    Returns the strongest spindle-band pole in each window.
    """
    if len(starts) == 0:
        return np.array([]), np.array([])

    def fit_one(s):
        return _fit_window(
            signal[s:s + window_samples],
            ar_order,
            target_fs,
            spindle_band,
        )

    if verbose:
        with tqdm_joblib(
            tqdm(total=len(starts), desc=desc, unit="win")
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

    r_vals = np.asarray(
        [r for r, _ in results],
        dtype=float,
    )
    f_vals = np.asarray(
        [f for _, f in results],
        dtype=float,
    )

    return r_vals, f_vals


def fit_ar_on_prepared_signal(
    signal,
    target_fs=256,
    ar_order=8,
    window_sec=1.0,
    stride_samples=1,
    spindle_band=(10, 15),
    n_jobs=1,
    verbose=False,
):
    """
    Backward-compatible public helper.

    NOTE:
    This helper intentionally retains the old one-pole-per-window output
    because its interface is used elsewhere in the application.

    The corrected spindle detector does NOT use this helper; it uses all
    poles internally.
    """
    window_samples = int(round(window_sec * target_fs))

    if len(signal) < window_samples:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    total_windows = len(signal) - window_samples + 1

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

    return r_vals, f_vals, starts


# ============================================================================
# All-pole fitting for the corrected detector
# ============================================================================

def _fit_all_poles_at_start(
    signal,
    start,
    window_samples,
    ar_order,
    target_fs,
):
    return _fit_window_all_poles(
        signal[start:start + window_samples],
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
            tqdm(total=len(starts), desc="AR fitting", unit="win")
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

    frequencies = [x[0] for x in results]
    radii = [x[1] for x in results]

    return frequencies, radii


# ============================================================================
# Paper-faithful pole candidates
# ============================================================================

def _get_spindle_candidates(
    frequencies,
    radii,
    spindle_band,
):
    """
    Return ALL spindle-band poles in one AR window.

    This is the key correction relative to the previous code:
    an AR(8) model can contain up to four positive-frequency oscillatory
    poles, and the papers do not say to discard all but the largest R pole.
    """
    if len(frequencies) == 0:
        return []

    low_f, high_f = spindle_band

    mask = (
        (frequencies >= low_f)
        & (frequencies <= high_f)
        & np.isfinite(frequencies)
        & np.isfinite(radii)
    )

    return [
        (float(f), float(r))
        for f, r in zip(
            frequencies[mask],
            radii[mask],
        )
    ]


# ============================================================================
# Frequency continuity
# ============================================================================

# The papers say that the lower threshold merges/splits detections
# "within a specific frequency", but do not define a numerical pole-matching
# rule. This internal value is therefore an implementation choice.
_DEFAULT_MAX_FREQUENCY_JUMP_HZ = 2.0


def _match_candidates_to_tracks(
    candidates,
    tracks,
    max_frequency_jump_hz,
):
    """
    Greedy nearest-frequency one-to-one matching.

    Each candidate is matched to at most one active track and vice versa.
    Unmatched candidates start new tracks.

    A track is a list of window observations:
        (window_index, start_sample, frequency, R)
    """
    if not tracks:
        return [], list(range(len(candidates)))

    pairs = []

    for track_id, track in enumerate(tracks):
        if not track["observations"]:
            continue

        last_f = track["observations"][-1][2]

        for cand_id, (cand_f, _) in enumerate(candidates):
            distance = abs(cand_f - last_f)
            if distance <= max_frequency_jump_hz:
                pairs.append((distance, track_id, cand_id))

    pairs.sort(key=lambda x: x[0])

    used_tracks = set()
    used_candidates = set()
    assignments = []

    for distance, track_id, cand_id in pairs:
        if track_id in used_tracks or cand_id in used_candidates:
            continue

        assignments.append((track_id, cand_id))
        used_tracks.add(track_id)
        used_candidates.add(cand_id)

    unmatched_candidates = [
        i for i in range(len(candidates))
        if i not in used_candidates
    ]

    return assignments, unmatched_candidates


# ============================================================================
# Event finalization
# ============================================================================

def _finalize_track_event(
    observations,
    target_fs,
    window_samples,
    events,
):
    """
    Convert one pole trajectory into the historical six-column output.

    Output:
        [start_time,
         peak_time,
         end_time,
         duration,
         max_r,
         peak_frequency]

    The event start/end are based on the upper-threshold crossings.
    Following Olbrich & Achermann, one window length is added to the
    duration to account for the 1-s temporal resolution.
    """
    if not observations:
        return

    observations = sorted(
        observations,
        key=lambda x: x[0],
    )

    rs = np.asarray(
        [x[3] for x in observations],
        dtype=float,
    )
    fs = np.asarray(
        [x[2] for x in observations],
        dtype=float,
    )
    starts = np.asarray(
        [x[1] for x in observations],
        dtype=float,
    )

    finite = (
        np.isfinite(rs)
        & np.isfinite(fs)
    )

    if not np.any(finite):
        return

    rs_valid = rs.copy()
    rs_valid[~finite] = -np.inf

    peak_i = int(np.argmax(rs_valid))

    max_r = float(rs[peak_i])
    peak_frequency = float(fs[peak_i])

    half_win = window_samples / (2.0 * target_fs)

    start_time = (
        starts[0] / target_fs
        + half_win
    )

    end_time = (
        starts[-1] / target_fs
        + half_win
    )

    peak_time = (
        starts[peak_i] / target_fs
        + half_win
    )

    # Olbrich & Achermann explicitly add td = 1 s.
    duration = (
        end_time
        - start_time
        + window_samples / target_fs
    )

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


# ============================================================================
# Multi-pole event detection
# ============================================================================

def _detect_events_all_poles(
    frequencies_per_window,
    radii_per_window,
    sample_starts,
    target_fs,
    window_samples,
    upper_threshold,
    lower_threshold,
    spindle_band,
    max_frequency_jump_hz=_DEFAULT_MAX_FREQUENCY_JUMP_HZ,
):
    """
    Detect spindle events from ALL spindle-band AR poles.

    Logic:
      1. Every spindle-band pole is considered independently.
      2. R > upper_threshold starts a detection.
      3. R >= lower_threshold keeps a pole trajectory continuous.
      4. The lower threshold therefore merges/splits nearby detections.
      5. Pole identity is maintained by nearest-frequency matching.
      6. At event finalization, the maximum R and its frequency are reported.
    """
    active_tracks = []
    completed_events = []

    def finalize_track(track):
        if track["active"]:
            _finalize_track_event(
                track["observations"],
                target_fs,
                window_samples,
                completed_events,
            )

    for win_i, (freqs, radii) in enumerate(
        zip(
            frequencies_per_window,
            radii_per_window,
        )
    ):
        candidates = _get_spindle_candidates(
            freqs,
            radii,
            spindle_band,
        )

        # Match this window's poles to existing trajectories.
        assignments, unmatched = _match_candidates_to_tracks(
            candidates,
            active_tracks,
            max_frequency_jump_hz,
        )

        matched_track_ids = set()

        # First, update matched tracks.
        for track_id, cand_id in assignments:
            track = active_tracks[track_id]
            f, r = candidates[cand_id]

            # If an active event falls below the lower threshold, it ends.
            if track["active"] and r < lower_threshold:
                finalize_track(track)
                track["active"] = False
                track["observations"] = []
                continue

            # Once an event has started, R >= lower_threshold keeps it alive.
            # Before onset, only R > upper_threshold can activate it.
            if not track["active"]:
                if r > upper_threshold:
                    track["active"] = True
                    track["observations"] = [
                        (
                            win_i,
                            sample_starts[win_i],
                            f,
                            r,
                        )
                    ]
            else:
                track["observations"].append(
                    (
                        win_i,
                        sample_starts[win_i],
                        f,
                        r,
                    )
                )

            matched_track_ids.add(track_id)

        # Existing tracks that were not matched disappear for this window.
        # A missing pole cannot be assumed to remain the same oscillator.
        for track_id, track in enumerate(active_tracks):
            if track_id not in matched_track_ids and track["active"]:
                finalize_track(track)
                track["active"] = False
                track["observations"] = []

        # Start new tracks from unmatched candidates only when they cross rb.
        for cand_id in unmatched:
            f, r = candidates[cand_id]

            active_tracks.append(
                {
                    "active": bool(r > upper_threshold),
                    "observations": (
                        [
                            (
                                win_i,
                                sample_starts[win_i],
                                f,
                                r,
                            )
                        ]
                        if r > upper_threshold
                        else []
                    ),
                }
            )

    # Finalize everything still active.
    for track in active_tracks:
        if track["active"]:
            finalize_track(track)

    # Keep events in chronological order.
    completed_events.sort(key=lambda x: x[0])

    return completed_events


# ============================================================================
# Duration filtering
# ============================================================================

def filter_events_by_duration(
    events_list,
    min_duration_sec=0.4,
    max_duration_sec=3.5,
):
    """
    Optional duration filtering.

    The AR detector itself does not require these filters.
    """
    if not events_list:
        return np.empty((0, 6), dtype=float)

    events = np.asarray(events_list, dtype=float)

    durations = events[:, 3]

    mask = np.ones(len(events), dtype=bool)

    if min_duration_sec is not None:
        mask &= durations >= min_duration_sec

    if max_duration_sec is not None:
        mask &= durations <= max_duration_sec

    return events[mask]


# ============================================================================
# o-Quality
# ============================================================================

def classify_o_quality(
    max_r,
    upper_threshold=0.92,
):
    """
    Classify an event using the Blanco-Duque et al. paper bins.

        0.92 <= r < 0.93 -> oQ1
        0.93 <= r < 0.94 -> oQ2
        0.94 <= r < 0.95 -> oQ3
        r >= 0.95        -> oQ4
    """
    if not np.isfinite(max_r):
        return None

    if max_r < upper_threshold:
        return None

    if max_r < 0.93:
        return "oQ1"
    if max_r < 0.94:
        return "oQ2"
    if max_r < 0.95:
        return "oQ3"

    return "oQ4"


# ============================================================================
# Main detector -- SAME PUBLIC INTERFACE
# ============================================================================

def find_spindles_lfp(
    raw_signal,
    fs,
    target_fs=256,
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
    Detect spindle-like events using the Olbrich/Blanco-Duque AR method.

    Public function signature is intentionally unchanged.

    Method
    ------
    1. Optional acquisition-style 0.1-100 Hz preprocessing.
    2. Resample to target_fs.
    3. Fit AR(8) independently to overlapping 1-s windows.
    4. Keep ALL positive-frequency poles in spindle_band.
    5. Detect each pole independently using upper/lower thresholds.
    6. Associate poles between adjacent windows by frequency continuity.
    7. Report one row per detected spindle event.

    IMPORTANT
    ---------
    The spindle detector itself does NOT band-pass filter to 10-15 Hz.
    The spindle band is applied to the AR pole frequencies after fitting.
    This is consistent with the paper's statement that the approach does
    not require filtering in a specific spindle band.

    Returns
    -------
    np.ndarray, shape (N, 6)

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
        raise ValueError("raw_signal must be one-dimensional.")

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

    if target_fs > fs:
        raise ValueError(
            "target_fs must not exceed the input sampling rate."
        )

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    #
    # Blanco-Duque et al. report that their acquired mouse EEG/LFP
    # signals were filtered 0.1-100 Hz and then resampled before AR
    # analysis. This preprocessing is retained here.
    #
    # It is NOT a spindle-band filter.
    #
    if lowcut is not None and highcut is not None:
        if lowcut <= 0:
            raise ValueError("lowcut must be > 0.")

        if highcut <= lowcut:
            raise ValueError(
                "highcut must be greater than lowcut."
            )

        if highcut >= fs / 2:
            raise ValueError(
                f"highcut ({highcut}) must be below input Nyquist "
                f"frequency ({fs / 2})."
            )

        filtered_signal = bandpass_filter(
            raw_signal,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
        )
    else:
        # Allows the caller to skip an additional filter when the data
        # have already undergone the acquisition-equivalent filtering.
        filtered_signal = raw_signal

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
        return np.empty((0, 6), dtype=float)

    total_windows = len(signal) - window_samples + 1

    starts = np.arange(
        0,
        total_windows,
        stride_samples,
    )

    # ------------------------------------------------------------
    # Fit ALL poles.
    # ------------------------------------------------------------
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

    if not frequencies_per_window:
        return np.empty((0, 6), dtype=float)

    # ------------------------------------------------------------
    # Multi-pole spindle detection.
    # ------------------------------------------------------------
    raw_events = _detect_events_all_poles(
        frequencies_per_window,
        radii_per_window,
        starts,
        target_fs,
        window_samples,
        upper_threshold,
        lower_threshold,
        spindle_band,
    )

    # ------------------------------------------------------------
    # Optional duration filtering.
    # ------------------------------------------------------------
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
