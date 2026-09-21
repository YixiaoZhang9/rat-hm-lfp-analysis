from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

# ============================================================
# Configuration
# ============================================================

@dataclass
class SpindleDetectorConfig:
    # Original recording
    input_fs: float = 1000.0

    # Paper preprocessing
    target_fs: float = 128.0
    highpass_hz: float = 0.1
    lowpass_hz: float = 100.0
    filter_order: int = 4

    # AR detector
    ar_order: int = 8
    window_seconds: float = 1.0
    step_samples: int = 1

    # Spindle frequency range
    spindle_low_hz: float = 10.0
    spindle_high_hz: float = 15.0

    # Paper thresholds
    upper_threshold: float = 0.92
    lower_threshold: float = 0.90

    # Used only for the implementation of pole tracking.
    # The paper does not specify a numerical frequency-jump criterion.
    max_frequency_jump_hz: float = 3.0

    # Numerical stability
    min_radius: float = 1e-8
    max_radius: float = 1.0 - 1e-10


# ============================================================
# Burg AR estimation
# ============================================================

def burg_ar(
    x: np.ndarray,
    order: int,
    demean: bool = True,
) -> np.ndarray:
    """
    Estimate AR coefficients using the Burg algorithm.

    Returns coefficients in the convention used by the paper:

        x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

    Therefore the AR polynomial is:

        z^p - a1*z^(p-1) - ... - ap = 0

    Parameters
    ----------
    x:
        1-D signal segment.
    order:
        AR model order.
    demean:
        Remove the segment mean before estimation.

    Returns
    -------
    a:
        Array [a1, ..., ap].
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("burg_ar expects a 1-D signal.")

    if len(x) <= order:
        raise ValueError(
            f"Signal segment ({len(x)} samples) is too short "
            f"for AR order {order}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")

    if demean:
        x = x - np.mean(x)

    # Forward and backward prediction errors.
    ef = x[1:].copy()
    eb = x[:-1].copy()

    # Standard AR polynomial convention:
    #
    # A(z) = 1 + c1 z^-1 + ... + cp z^-p
    #
    # The paper uses:
    #
    # x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]
    #
    # so a_i = -c_i.
    c = np.zeros(order + 1, dtype=float)
    c[0] = 1.0

    eps = np.finfo(float).eps

    for m in range(1, order + 1):

        denominator = np.dot(ef, ef) + np.dot(eb, eb)

        if denominator <= eps:
            break

        reflection = -2.0 * np.dot(ef, eb) / denominator

        # Numerical protection.
        reflection = float(
            np.clip(
                reflection,
                -0.999999999999,
                0.999999999999,
            )
        )

        c_old = c[:m].copy()

        c[m] = reflection

        if m > 1:
            c[1:m] = (
                c_old[1:m]
                + reflection * c_old[m - 1:0:-1]
            )

        ef_old = ef.copy()
        eb_old = eb.copy()

        ef = ef_old[1:] + reflection * eb_old[1:]
        eb = eb_old[:-1] + reflection * ef_old[:-1]

    # Convert to paper convention.
    a = -c[1:]

    return a


# ============================================================
# AR coefficients -> poles -> frequency/radius
# ============================================================

def ar_coefficients_to_poles(
    ar_coefficients: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert AR coefficients to pole frequencies and radii.

    Paper equation:

        z^p - sum_k a_k z^(p-k) = product_k (z-z_k)

    For z_k = r_k * exp(i*phi_k):

        f_k = phi_k / (2*pi*Delta)

    with Delta = 1/fs.

    Returns
    -------
    frequencies:
        Positive-frequency pole frequencies in Hz.

    radii:
        Corresponding pole radii.
    """
    a = np.asarray(ar_coefficients, dtype=float)
    p = len(a)

    # z^p - a1*z^(p-1) - ... - ap
    polynomial = np.concatenate(
        ([1.0], -a)
    )

    roots = np.roots(polynomial)

    radii = np.abs(roots)
    phases = np.angle(roots)

    frequencies = phases * fs / (2.0 * np.pi)

    # Keep only positive-frequency members of conjugate pairs.
    mask = frequencies > 0.0

    frequencies = frequencies[mask]
    radii = radii[mask]

    # Sort by frequency.
    order = np.argsort(frequencies)

    frequencies = frequencies[order]
    radii = radii[order]

    return frequencies, radii


def estimate_window_modes(
    segment: np.ndarray,
    fs: float,
    ar_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit AR model to one window and return all positive-frequency
    oscillatory poles.
    """
    coefficients = burg_ar(
        segment,
        order=ar_order,
    )

    frequencies, radii = ar_coefficients_to_poles(
        coefficients,
        fs=fs,
    )

    return frequencies, radii


# ============================================================
# Preprocessing
# ============================================================

def bandpass_preprocess(
    signal: np.ndarray,
    fs: float,
    highpass_hz: float,
    lowpass_hz: float,
    filter_order: int,
) -> np.ndarray:
    """
    Apply the broad 0.1-100 Hz preprocessing described in the paper.

    The paper specifies the frequency range but does not specify
    the exact filter type/order for this preprocessing step.

    Here we use a zero-phase Butterworth SOS filter.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be 1-D.")

    nyquist = fs / 2.0

    if lowpass_hz >= nyquist:
        raise ValueError(
            f"Low-pass frequency ({lowpass_hz} Hz) must be below "
            f"Nyquist ({nyquist} Hz)."
        )

    sos = butter(
        filter_order,
        [
            highpass_hz / nyquist,
            lowpass_hz / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def resample_to_128(
    signal: np.ndarray,
    input_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Polyphase resampling.

    For 1000 -> 128 Hz this becomes:

        up = 16
        down = 125

    giving exactly 128 Hz.
    """
    if input_fs == target_fs:
        return np.asarray(signal, dtype=float)

    ratio = np.gcd(
        int(round(input_fs)),
        int(round(target_fs)),
    )

    up = int(round(target_fs)) // ratio
    down = int(round(input_fs)) // ratio

    return resample_poly(
        np.asarray(signal, dtype=float),
        up,
        down,
    )


# ============================================================
# Window-by-window AR analysis
# ============================================================

def compute_ar_timeseries(
    signal_128: np.ndarray,
    config: SpindleDetectorConfig,
) -> pd.DataFrame:
    """
    Run AR(8) on every overlapping 1-second window.

    Windows:
        length = 128 samples
        shift = 1 sample

    Returns one row per AR window.

    Columns:
        time
        mode_1_frequency
        mode_1_radius
        ...
        mode_4_frequency
        mode_4_radius
    """
    signal_128 = np.asarray(signal_128, dtype=float)

    fs = config.target_fs

    window_samples = int(
        round(config.window_seconds * fs)
    )

    if len(signal_128) < window_samples:
        return pd.DataFrame()

    n_windows = (
        1
        + (len(signal_128) - window_samples)
        // config.step_samples
    )

    rows = []

    for window_idx in range(n_windows):

        start = (
            window_idx
            * config.step_samples
        )

        stop = start + window_samples

        segment = signal_128[start:stop]

        try:
            frequencies, radii = estimate_window_modes(
                segment,
                fs=fs,
                ar_order=config.ar_order,
            )
        except (ValueError, np.linalg.LinAlgError):
            frequencies = np.array([])
            radii = np.array([])

        row = {
            "window_index": window_idx,
            "start_sample": start,
            "time": start / fs,
        }

        # AR(8) gives at most four positive-frequency
        # oscillatory modes.
        for mode in range(4):

            if mode < len(frequencies):
                row[f"mode_{mode + 1}_frequency"] = frequencies[mode]
                row[f"mode_{mode + 1}_radius"] = radii[mode]
            else:
                row[f"mode_{mode + 1}_frequency"] = np.nan
                row[f"mode_{mode + 1}_radius"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Event extraction
# ============================================================

def assign_oq(r_max: float) -> str:
    """
    Paper o-Quality categories.
    """
    if 0.92 <= r_max < 0.93:
        return "oQ1"
    elif 0.93 <= r_max < 0.94:
        return "oQ2"
    elif 0.94 <= r_max < 0.95:
        return "oQ3"
    elif r_max >= 0.95:
        return "oQ4"

    return ""


def _extract_events_from_mode(
    times: np.ndarray,
    frequencies: np.ndarray,
    radii: np.ndarray,
    config: SpindleDetectorConfig,
) -> List[Dict]:
    """
    Extract spindle events from one tracked AR mode.

    Important:
    - rb = 0.92 starts an event.
    - ra = 0.90 is used as the hysteresis/merge threshold.
    - A temporary drop below rb but remaining >= ra does not
      immediately split the event.
    - A sustained drop below ra terminates the event.
    """
    events = []

    active = False
    start_idx = None

    # Last sample at which the mode was above the detection
    # threshold.
    last_above_rb_idx = None

    # Store all samples belonging to current event.
    event_indices = []

    # Frequency continuity state.
    previous_frequency = np.nan

    for i in range(len(times)):

        f = frequencies[i]
        r = radii[i]

        valid_spindle = (
            np.isfinite(f)
            and np.isfinite(r)
            and config.spindle_low_hz <= f <= config.spindle_high_hz
        )

        if not valid_spindle:
            if active:
                # Invalid frequency breaks the mode.
                if last_above_rb_idx is not None:
                    events.append(
                        _finalize_event(
                            event_indices,
                            times,
                            frequencies,
                            radii,
                            last_above_rb_idx,
                            config,
                        )
                    )

                active = False
                start_idx = None
                last_above_rb_idx = None
                event_indices = []

            previous_frequency = np.nan
            continue

        # ----------------------------------------------------
        # Not currently in an event.
        # ----------------------------------------------------
        if not active:

            if r > config.upper_threshold:

                active = True
                start_idx = i
                last_above_rb_idx = i
                event_indices = [i]

                previous_frequency = f

            continue

        # ----------------------------------------------------
        # Currently tracking an event.
        # ----------------------------------------------------

        # If frequency changes unrealistically abruptly,
        # consider this a different oscillatory mode.
        if (
            np.isfinite(previous_frequency)
            and abs(f - previous_frequency)
            > config.max_frequency_jump_hz
        ):
            if last_above_rb_idx is not None:
                events.append(
                    _finalize_event(
                        event_indices,
                        times,
                        frequencies,
                        radii,
                        last_above_rb_idx,
                        config,
                    )
                )

            active = False
            start_idx = None
            last_above_rb_idx = None
            event_indices = []

            # This new point can start another event.
            if r > config.upper_threshold:
                active = True
                start_idx = i
                last_above_rb_idx = i
                event_indices = [i]

            previous_frequency = f
            continue

        event_indices.append(i)
        previous_frequency = f

        if r > config.upper_threshold:
            last_above_rb_idx = i

        # A fall below the lower threshold separates events.
        if r < config.lower_threshold:

            if last_above_rb_idx is not None:

                events.append(
                    _finalize_event(
                        event_indices,
                        times,
                        frequencies,
                        radii,
                        last_above_rb_idx,
                        config,
                    )
                )

            active = False
            start_idx = None
            last_above_rb_idx = None
            event_indices = []

            previous_frequency = np.nan

    # Finish event at recording boundary.
    if active and last_above_rb_idx is not None:
        events.append(
            _finalize_event(
                event_indices,
                times,
                frequencies,
                radii,
                last_above_rb_idx,
                config,
            )
        )

    return events


def _finalize_event(
    indices: List[int],
    times: np.ndarray,
    frequencies: np.ndarray,
    radii: np.ndarray,
    end_idx: int,
    config: SpindleDetectorConfig,
) -> Dict:
    """
    Convert an event's window indices into an event record.

    The event's max-r sample determines:
        - max_r
        - frequency_at_max_r
    """
    if not indices:
        raise ValueError("Cannot finalize an empty event.")

    indices = np.asarray(indices, dtype=int)

    event_radii = radii[indices]

    finite = np.isfinite(event_radii)

    if not np.any(finite):
        raise ValueError("Event has no finite radius values.")

    finite_indices = indices[finite]

    max_local = np.argmax(
        radii[finite_indices]
    )

    max_idx = finite_indices[max_local]

    start_idx = indices[0]

    # The paper defines t2 from the point where r falls
    # below rb. Therefore use the last above-rb window.
    final_end_idx = min(
        end_idx,
        len(times) - 1,
    )

    r_max = float(radii[max_idx])
    f_at_max = float(frequencies[max_idx])

    start_time = float(times[start_idx])

    # A 1-s AR window represents an interval, not a point.
    # We report the end of the final 1-s window.
    window_duration = config.window_seconds

    end_time = float(
        times[final_end_idx]
        + window_duration
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "max_r": r_max,
        "frequency_at_max_r": f_at_max,
        "o_quality": assign_oq(r_max),
    }


def extract_spindle_events(
    ar_df: pd.DataFrame,
    config: SpindleDetectorConfig,
) -> pd.DataFrame:
    """
    Extract spindle events from the four AR oscillatory modes.

    Modes are sorted by frequency independently in each AR window.
    This is an implementation choice because the paper does not
    specify an exact pole-tracking algorithm.
    """
    if ar_df.empty:
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "duration",
                "max_r",
                "frequency_at_max_r",
                "o_quality",
                "mode",
            ]
        )

    all_events = []

    times = ar_df["time"].to_numpy()

    for mode in range(1, 5):

        f_col = f"mode_{mode}_frequency"
        r_col = f"mode_{mode}_radius"

        frequencies = ar_df[f_col].to_numpy()
        radii = ar_df[r_col].to_numpy()

        events = _extract_events_from_mode(
            times,
            frequencies,
            radii,
            config,
        )

        for event in events:
            event["mode"] = mode
            all_events.append(event)

    if not all_events:
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "duration",
                "max_r",
                "frequency_at_max_r",
                "o_quality",
                "mode",
            ]
        )

    events_df = pd.DataFrame(all_events)

    return events_df.sort_values(
        "start_time"
    ).reset_index(drop=True)


# ============================================================
# MATLAB file loading
# ============================================================

def list_mat_variables(path: Union[str, Path]) -> List[str]:
    """
    List candidate variables in a MATLAB .mat file.

    Supports normal MAT files and MATLAB v7.3/HDF5 files.
    """
    path = Path(path)

    try:
        data = loadmat(
            path,
            squeeze_me=False,
            struct_as_record=False,
        )

        return [
            key
            for key, value in data.items()
            if not key.startswith("__")
            and isinstance(value, np.ndarray)
        ]

    except NotImplementedError:
        # MATLAB v7.3
        with h5py.File(path, "r") as f:
            return list(f.keys())


def load_mat_signal(
    path: Union[str, Path],
    variable: str,
    channel: int = 0,
    channel_axis: Union[int, str] = "auto",
) -> np.ndarray:
    """
    Load one EEG/LFP signal from a MATLAB .mat file.

    Parameters
    ----------
    path:
        .mat file.
    variable:
        Name of the MATLAB variable containing the signal.
    channel:
        Channel index if the variable contains multiple channels.
    channel_axis:
        0, 1, or "auto".

    "auto" assumes the smaller dimension is the channel dimension.
    This is convenient but should be checked against your data.

    Returns
    -------
    signal:
        1-D numpy array.
    """
    path = Path(path)

    try:
        mat = loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
        )

        if variable not in mat:
            raise KeyError(
                f"Variable '{variable}' not found in {path}.\n"
                f"Available variables: "
                f"{list_mat_variables(path)}"
            )

        data = np.asarray(mat[variable])

    except NotImplementedError:
        # MATLAB v7.3
        with h5py.File(path, "r") as f:
            if variable not in f:
                raise KeyError(
                    f"Variable '{variable}' not found in {path}.\n"
                    f"Available variables: {list(f.keys())}"
                )

            data = np.asarray(f[variable])

    data = np.squeeze(data)

    if data.ndim == 1:
        return data.astype(float)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 1-D or 2-D signal array, got "
            f"shape {data.shape}."
        )

    if channel_axis == "auto":

        # Usually one dimension is number of channels and the
        # other is number of samples.
        if data.shape[0] <= data.shape[1]:
            channel_axis = 0
        else:
            channel_axis = 1

    if channel_axis == 0:
        if channel >= data.shape[0]:
            raise IndexError(
                f"Channel {channel} unavailable for shape {data.shape}."
            )

        signal = data[channel, :]

    elif channel_axis == 1:
        if channel >= data.shape[1]:
            raise IndexError(
                f"Channel {channel} unavailable for shape {data.shape}."
            )

        signal = data[:, channel]

    else:
        raise ValueError(
            "channel_axis must be 0, 1, or 'auto'."
        )

    return np.asarray(
        signal,
        dtype=float,
    ).squeeze()


# ============================================================
# Main detector
# ============================================================

class ARSpindleDetector:
    """
    Offline AR-based spindle detector following
    Blanco-Duque et al. (2024).
    """

    def __init__(
        self,
        config: Optional[SpindleDetectorConfig] = None,
    ):
        self.config = (
            config
            if config is not None
            else SpindleDetectorConfig()
        )

    def preprocess(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        0.1-100 Hz filtering followed by resampling to 128 Hz.
        """
        signal = bandpass_preprocess(
            signal,
            fs=self.config.input_fs,
            highpass_hz=self.config.highpass_hz,
            lowpass_hz=self.config.lowpass_hz,
            filter_order=self.config.filter_order,
        )

        signal = resample_to_128(
            signal,
            input_fs=self.config.input_fs,
            target_fs=self.config.target_fs,
        )

        return signal

    def detect_signal(
        self,
        signal: np.ndarray,
        return_ar_timeseries: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Detect spindles in one signal.

        Returns
        -------
        events_df:
            One row per detected spindle.

        ar_df:
            Window-by-window AR frequency/radius information.
        """
        processed = self.preprocess(signal)

        ar_df = compute_ar_timeseries(
            processed,
            config=self.config,
        )

        events_df = extract_spindle_events(
            ar_df,
            config=self.config,
        )

        if not return_ar_timeseries:
            ar_df = None

        return events_df, ar_df

    def detect_task(
        self,
        task: Dict,
        signal_variable: str,
        channel: int = 0,
        channel_axis: Union[int, str] = "auto",
        return_ar_timeseries: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run detection on one TaskLoader task.
        """
        signal = load_mat_signal(
            task["data_path"],
            variable=signal_variable,
            channel=channel,
            channel_axis=channel_axis,
        )

        events_df, ar_df = self.detect_signal(
            signal,
            return_ar_timeseries=return_ar_timeseries,
        )

        # Add manifest metadata.
        metadata_columns = [
            "cohort",
            "rat",
            "region",
            "date",
            "file_name",
            "data_path",
            "scoring_path",
        ]

        for column in metadata_columns:
            if column in task:
                events_df[column] = task[column]

            if ar_df is not None and column in task:
                ar_df[column] = task[column]

        return events_df, ar_df

    def detect_tasks(
        self,
        tasks: Sequence[Dict],
        signal_variable: str,
        channel: int = 0,
        channel_axis: Union[int, str] = "auto",
        return_ar_timeseries: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run detection over all tasks produced by TaskLoader.
        """
        all_events = []
        all_ar = []

        for i, task in enumerate(tasks):

            print(
                f"[{i + 1}/{len(tasks)}] "
                f"{task.get('rat', '')} "
                f"{task.get('region', '')} "
                f"{task.get('date', '')}"
            )

            events_df, ar_df = self.detect_task(
                task,
                signal_variable=signal_variable,
                channel=channel,
                channel_axis=channel_axis,
                return_ar_timeseries=return_ar_timeseries,
            )

            all_events.append(events_df)

            if return_ar_timeseries and ar_df is not None:
                all_ar.append(ar_df)

        if all_events:
            events = pd.concat(
                all_events,
                ignore_index=True,
            )
        else:
            events = pd.DataFrame()

        if return_ar_timeseries and all_ar:
            ar = pd.concat(
                all_ar,
                ignore_index=True,
            )
        else:
            ar = None

        return events, ar
