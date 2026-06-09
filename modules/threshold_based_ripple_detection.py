"""
Ripple Detection Methods
========================

This module implements several hippocampal ripple detection methods used in the literature.

Implemented methods
-------------------
- Karlsson & Frank (2009)
- Diba & Buzsáki (2007)
- Dahal et al. (2023)
- Vicente et al. (2020)

========================
"""

# Imports
import numpy as np
from scipy.signal import (
    butter,
    convolve,
    filtfilt,
    get_window,
    hilbert,
)
from scipy.signal.windows import gaussian
from scipy.stats import zscore


# Functions
def filter_lfp(
    lfp: np.ndarray,
    fs: float,
    freq_range: tuple = (100, 250),
) -> np.ndarray:
    """
    Bandpass filter LFP signal.

    Parameters
    ----------
    lfp : np.ndarray
        Input LFP signal.
    fs : float
        Sampling frequency in Hz.
    freq_range : tuple, optional
        Frequency range for filtering in Hz.

    Returns
    -------
    np.ndarray
        Filtered LFP signal.
    """
    b, a = butter(
        4,
        np.array(freq_range) / (fs / 2),
        btype="band",
    )
    return filtfilt(b, a, lfp)


def smooth_signal(
    signal: np.ndarray,
    fs: float,
    sigma: float,
) -> np.ndarray:
    """
    Smooth signal using a Gaussian kernel.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.
    sigma : float
        Gaussian smoothing sigma in seconds.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    smoothing_sigma = sigma * fs
    window_size = int(smoothing_sigma * 6)
    if window_size % 2 == 0:
        window_size += 1
    gauss_filter = gaussian(
        window_size,
        smoothing_sigma,
    )
    gauss_filter /= np.sum(gauss_filter)
    smoothed_signal = convolve(
        signal,
        gauss_filter,
        mode="same",
    )
    return smoothed_signal


def merge_close_ripples(
    start_idx,
    end_idx,
    fs,
    close_ripple_thresh,
):
    """
    Merge ripple events that are temporally close.

    Parameters
    ----------
    start_idx : array-like
        Ripple start indices.
    end_idx : array-like
        Ripple end indices.
    fs : float
        Sampling frequency in Hz.
    close_ripple_thresh : float
        Maximum inter-ripple interval for merging in seconds.

    Returns
    -------
    np.ndarray
        Merged ripple start indices.
    np.ndarray
        Merged ripple end indices.
    """
    start_idx = np.array(start_idx)
    end_idx = np.array(end_idx)

    i = 0

    while i < len(start_idx) - 1:
        interval_duration = (start_idx[i + 1] - end_idx[i]) / fs

        if interval_duration <= close_ripple_thresh:
            start_idx[i + 1] = start_idx[i]
            start_idx = np.delete(start_idx, i)
            end_idx = np.delete(end_idx, i)

        else:
            i += 1

    return start_idx, end_idx


def fastrms(
    signal,
    window_size=5,
    apply_amplitude_correction=False,
):
    """
    Compute moving RMS power.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int, optional
        RMS window size in samples.
    apply_amplitude_correction : bool, optional
        Apply sinusoidal amplitude correction.

    Returns
    -------
    np.ndarray
        RMS signal.
    """
    if signal.ndim > 2:
        raise ValueError("Input signal must be 1D or 2D.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    window = get_window(
        "boxcar",
        window_size,
    )

    power = signal**2

    if signal.ndim == 1:
        rms = convolve(
            power,
            window,
            mode="same",
        )

    else:
        rms = np.array(
            [
                convolve(
                    power[:, col],
                    window,
                    mode="same",
                )
                for col in range(signal.shape[1])
            ]
        ).T

    rms = np.sqrt(rms / np.sum(window))
    if apply_amplitude_correction:
        rms *= np.sqrt(2)

    return rms


# Ripple Detection Methods


def find_ripples_karlsson(
    lfp,
    fs,
    min_duration=0.015,
    zscore_thresh=3,
    smoothing_sigma=0.004,
):
    """
    Detect ripples using the Karlsson & Frank method.

    References
    ----------
    .. [1] Karlsson, M.P., & Frank, L.M. (2009).
           Awake replay of remote experiences in the hippocampus.
           Nature Neuroscience, 12, 913–918.
    """
    filtered_lfp = filter_lfp(
        lfp,
        fs,
        (100, 250),
    )

    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))

    smoothed_envelope = smooth_signal(
        instantaneous_amplitude,
        fs,
        smoothing_sigma,
    )

    zscored_envelope = zscore(smoothed_envelope)

    above_mean = zscored_envelope > 0

    above_thresh = zscored_envelope > zscore_thresh

    threshold_envelope = zscore_thresh * np.std(smoothed_envelope) + np.mean(
        smoothed_envelope
    )

    start_idx_mean = np.where(
        np.diff(np.concatenate(([0], above_mean.astype(int)))) == 1
    )[0]

    end_idx_mean = np.where(
        np.diff(np.concatenate((above_mean.astype(int), [0]))) == -1
    )[0]

    start_idx_thresh = np.where(
        np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1
    )[0]

    end_idx_thresh = np.where(
        np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ripple = []
    end_idx_ripple = []
    peak_idx_ripple = []

    for start, end in zip(
        start_idx_thresh,
        end_idx_thresh,
    ):
        in_mean_idx = np.where((start_idx_mean <= start) & (end_idx_mean >= end))[0]

        if len(in_mean_idx) == 0:
            continue

        idx = in_mean_idx[0]

        if (end - start) / fs < min_duration:
            continue

        ripple_start = start_idx_mean[idx]
        ripple_end = end_idx_mean[idx]

        start_idx_ripple.append(ripple_start)
        end_idx_ripple.append(ripple_end)

        segment = filtered_lfp[ripple_start:ripple_end]

        peak_local = np.argmax(segment)

        peak_idx_ripple.append(ripple_start + peak_local)

    start_idx_ripple = np.unique(start_idx_ripple)

    end_idx_ripple = np.unique(end_idx_ripple)

    peak_idx_ripple = np.unique(peak_idx_ripple)

    return {
        "start_idx": start_idx_ripple,
        "end_idx": end_idx_ripple,
        "peak_idx": peak_idx_ripple,
        "start_time": start_idx_ripple / fs,
        "end_time": end_idx_ripple / fs,
        "peak_time": peak_idx_ripple / fs,
    }


def find_ripples_dahal(
    lfp,
    fs,
    min_duration=0.02,
    zscore_thresh=2.0,
    zscore_peak_thresh=5.0,
    smoothing_sigma=0.004,
    close_ripple_thresh=0.03,
):
    """
    Detect ripples using the Dahal et al. method.

    References
    ----------
    .. [1] Dahal, P. et al. (2023).
           Hippocampal–cortical coupling differentiates
           long-term memory processes.
           PNAS.
    """
    filtered_lfp = filter_lfp(
        lfp,
        fs,
        (100, 250),
    )

    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))

    smoothed_envelope = smooth_signal(
        instantaneous_amplitude,
        fs,
        smoothing_sigma,
    )

    zscored_envelope = zscore(smoothed_envelope)

    above_thresh = zscored_envelope > zscore_thresh

    start_idx_thresh = np.where(
        np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1
    )[0]

    end_idx_thresh = np.where(
        np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ripple = []
    end_idx_ripple = []

    for start, end in zip(
        start_idx_thresh,
        end_idx_thresh,
    ):
        peak_ripple = np.max(zscored_envelope[start:end])

        duration_ripple = (end - start) / fs

        if peak_ripple > zscore_peak_thresh and duration_ripple > min_duration:
            start_idx_ripple.append(start)
            end_idx_ripple.append(end)

    start_idx_ripple, end_idx_ripple = merge_close_ripples(
        start_idx_ripple,
        end_idx_ripple,
        fs,
        close_ripple_thresh,
    )

    start_idx_ripple = np.array(start_idx_ripple)

    end_idx_ripple = np.array(end_idx_ripple)

    return {
        "start_idx": start_idx_ripple,
        "end_idx": end_idx_ripple,
        "peak_idx": None,
        "start_time": start_idx_ripple / fs,
        "end_time": end_idx_ripple / fs,
        "peak_time": None,
    }


def find_ripples_diba(
    lfp,
    fs,
    min_duration=0,
    zscore_thresh=2,
):
    """
    Detect ripples using the Diba & Buzsáki method.
    """
    filtered_lfp = filter_lfp(
        lfp,
        fs,
        (100, 250),
    )

    window_size = int(0.005 * fs)

    power = fastrms(
        filtered_lfp,
        window_size,
    )

    mean_power = np.mean(power)
    std_power = np.std(power)

    above_thresh = power > (mean_power + zscore_thresh * std_power)

    above_extension_thresh = power > (mean_power + 1.5 * std_power)

    start_idx_thresh = np.where(
        np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1
    )[0]

    end_idx_thresh = np.where(
        np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ext = np.where(
        np.diff(np.concatenate(([0], above_extension_thresh.astype(int)))) == 1
    )[0]

    end_idx_ext = np.where(
        np.diff(np.concatenate((above_extension_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ripple = []
    end_idx_ripple = []

    for start, end in zip(
        start_idx_thresh,
        end_idx_thresh,
    ):
        in_mean_idx = np.where((start_idx_ext <= start) & (end_idx_ext >= end))[0]

        if len(in_mean_idx) == 0:
            continue

        idx = in_mean_idx[0]

        ripple_start = start_idx_ext[idx]
        ripple_end = end_idx_ext[idx]

        if (ripple_end - ripple_start) / fs >= min_duration:
            start_idx_ripple.append(ripple_start)

            end_idx_ripple.append(ripple_end)

    start_idx_ripple = np.unique(start_idx_ripple)

    end_idx_ripple = np.unique(end_idx_ripple)

    return {
        "start_idx": start_idx_ripple,
        "end_idx": end_idx_ripple,
        "peak_idx": None,
        "start_time": start_idx_ripple / fs,
        "end_time": end_idx_ripple / fs,
        "peak_time": None,
    }


def find_ripples_vicente(
    lfp,
    fs,
    min_duration=0.015,
    zscore_thresh=5,
    close_ripple_thresh=0.015,
):
    """
    Detect ripples using the Vicente et al. method.

    References
    ----------
    .. [1] Vicente, A.F. et al. (2020).
           In vivo characterization of neurophysiological
           diversity in the lateral supramammillary nucleus
           during hippocampal sharp-wave ripples.
           Neuroscience, 435, 95–111.
    """
    filtered_lfp = filter_lfp(
        lfp,
        fs,
        (100, 250),
    )

    window_size = int(0.005 * fs)

    power = fastrms(
        filtered_lfp,
        window_size,
    )

    mean_power = np.mean(power)
    std_power = np.std(power)

    above_thresh = power > (mean_power + zscore_thresh * std_power)

    above_extension_thresh = power > (mean_power + 0.5 * std_power)

    start_idx_thresh = np.where(
        np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1
    )[0]

    end_idx_thresh = np.where(
        np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ext = np.where(
        np.diff(np.concatenate(([0], above_extension_thresh.astype(int)))) == 1
    )[0]

    end_idx_ext = np.where(
        np.diff(np.concatenate((above_extension_thresh.astype(int), [0]))) == -1
    )[0]

    start_idx_ripple = []
    end_idx_ripple = []

    for start, end in zip(
        start_idx_thresh,
        end_idx_thresh,
    ):
        in_mean_idx = np.where((start_idx_ext <= start) & (end_idx_ext >= end))[0]

        if len(in_mean_idx) == 0:
            continue

        idx = in_mean_idx[0]

        ripple_start = start_idx_ext[idx]
        ripple_end = end_idx_ext[idx]

        if (ripple_end - ripple_start) / fs >= min_duration:
            start_idx_ripple.append(ripple_start)

            end_idx_ripple.append(ripple_end)

    start_idx_ripple, end_idx_ripple = merge_close_ripples(
        start_idx_ripple,
        end_idx_ripple,
        fs,
        close_ripple_thresh,
    )

    start_idx_ripple = np.array(start_idx_ripple)

    end_idx_ripple = np.array(end_idx_ripple)

    return {
        "start_idx": start_idx_ripple,
        "end_idx": end_idx_ripple,
        "peak_idx": None,
        "start_time": start_idx_ripple / fs,
        "end_time": end_idx_ripple / fs,
        "peak_time": None,
    }
