import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, get_window, hilbert

from modules.lfp_artifact_MAD_detection import mad_artifact_detector


def intervals_to_mask(intervals, timestamps):
    mask = np.zeros_like(timestamps, dtype=bool)
    for start, end in intervals:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def compute_peak_mean_frequency(segment, fs, window_type="hamming", freq_band=(9, 20)):
    """
    Compute the spectral centroid of a short signal segment.

    Parameters
    ----------
    segment : array_like
        1D array representing the signal segment
    fs : float
        Sampling frequency in Hz.
    window_type : str, optional
        Type of window to apply before FFT (e.g., 'hamming', 'hanning', 'blackman'). Default is 'hamming'.

    Returns
    -------
    peak_freq : float
        Frequency with maximum power within freq_band.
    mean_freq : float
        Power-weighted mean frequency within freq_band.
    """

    segment = np.asarray(segment).flatten()
    L = len(segment)
    segment = segment - np.mean(segment)

    # Apply window
    window = get_window(window_type, L)
    xw = segment * window

    # Determine FFT length
    # use at least 256 points and otherwise the next power of 2
    # greater than or equal to the ripple length.
    # This provides a denser frequency grid for frequency estimation
    # while preserving all samples of the ripple event.
    n_fft = max(256, 2 ** int(np.ceil(np.log2(L))))

    # Compute one-sided FFT
    X = np.fft.rfft(xw, n=n_fft)
    P = np.abs(X) ** 2  # power spectrum
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    # only in ripple frequency band
    fmin, fmax = freq_band
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return np.nan, np.nan

    freqs_band = freqs[mask]
    P_band = P[mask]

    if np.sum(P_band) == 0:
        return np.nan, np.nan

    # Peak frequency
    peak_freq = freqs_band[np.argmax(P_band)]

    # Mean frequency / spectral centroid
    mean_freq = np.sum(freqs_band * P_band) / np.sum(P_band)

    return peak_freq, mean_freq


def find_spindles_lfp_wavelet(
    raw_signal,
    signal,
    fs,
    freq_range=(9, 20),
    duration_core_min=0.2,
    duration_min=0.4,
    duration_max=3.5,
    merge_interval=0.5,
    amp_core=6,
    amp=3,
    threshold_type="median",
):
    """
    Detect sleep spindles using a wavelet-like spindle-band power detector with artifact rejection.

    Parameters
    ----------
    raw_signal : ndarray, shape (n_samples,)
        Raw LFP signal used for artifact detection.

    signal : ndarray, shape (n_samples,)
        Spindle-band filtered LFP signal, typically filtered between
        9 and 20 Hz.

    fs : float
        Sampling frequency in Hz.

    freq_range : tuple of float, optional
        Frequency range used for spindle-band power estimation.

        Default is (9, 20).

    duration_core_min : float, optional
        Minimum duration of the high-threshold spindle core in seconds.

        Default is 0.2.

    duration_min : float, optional
        Minimum allowed spindle duration in seconds.

        Default is 0.4.

    duration_max : float, optional
        Maximum allowed spindle duration in seconds.

        Default is 3.5.

    merge_interval : float, optional
        Maximum temporal gap in seconds for merging nearby spindle events.

        Default is 0.5.

    amp_core : float, optional
        Multiplicative threshold for detecting spindle core periods.

        Default is 6.

    amp : float, optional
        Multiplicative threshold for defining spindle event boundaries.

        Default is 3.

    threshold_type : {"median", "mean"}, optional
        Method used to estimate baseline band power from artifact-free
        samples.

        Default is "median".

    Returns
    -------
    spindles : ndarray, shape (n_events, 7)
        Detected spindle events and features. Columns are:

        0. start index
        1. peak index
        2. end index
        3. duration in seconds
        4. peak Hilbert-envelope amplitude
        5. FFT-based peak frequency in Hz within `freq_range`
        6. FFT-based power-weighted mean frequency in Hz within `freq_range`

        If no spindle is detected, returns an empty array with shape
        ``(0, 7)``.
    """

    n_samples = len(signal)

    # Detect LFP artifacts using the median absolute deviation method
    (valid_times, artifact_intervals_s) = mad_artifact_detector(
        raw_signal,
        mad_thresh=6.0,
        proportion_above_thresh=0.1,
        removal_window_ms=100.0,
        sampling_frequency=fs,
    )
    t = np.arange(len(signal)) / fs
    valid_mask = intervals_to_mask(valid_times, t)

    # extract the envelope
    envelope = np.abs(hilbert(signal))

    # Construct band wavelet
    npoints = 8001
    hz = np.fft.fftfreq(npoints, d=1 / fs)

    low, high = freq_range
    center = (low + high) / 2
    width = (high - low) / 2

    fx = np.exp(-0.5 * ((np.abs(hz) - center) / width) ** 2)

    wavelet = np.real(np.fft.fftshift(np.fft.ifft(fx)))
    wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))  # L2 normalization

    # apply Convolution
    conv = fftconvolve(signal, wavelet, mode="same")
    coef = np.abs(conv) ** 2

    #  smoothing
    sigma_sec = 0.04
    sigma_samples = sigma_sec * fs
    coef_smooth = gaussian_filter1d(coef, sigma=sigma_samples)

    x = coef_smooth
    # Robust threshold estimation
    # only use clean regions
    valid_x = x[valid_mask]

    if len(valid_x) == 0:
        return np.empty((0, 7))

    if threshold_type == "median":
        threshold = np.median(valid_x)
    else:
        threshold = np.mean(valid_x)

    if threshold <= 0:
        return np.empty((0, 7))

    over = (x > amp * threshold) & valid_mask
    core = (x > amp_core * threshold) & valid_mask

    def get_segments(mask):
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        start = np.where(diff == 1)[0]
        end = np.where(diff == -1)[0] - 1
        return start, end

    start_idx_core_spindle, end_idx_core_spindle = get_segments(core)
    start_idx_spindle, end_idx_spindle = get_segments(over)

    events = []

    for start_core, end_core in zip(start_idx_core_spindle, end_idx_core_spindle):
        duration_core = (end_core - start_core + 1) / fs
        if not (duration_core_min <= duration_core <= duration_max):
            continue

        start_idx = start_idx_spindle[start_idx_spindle <= start_core]
        end_idx = end_idx_spindle[end_idx_spindle >= end_core]

        if len(start_idx) == 0 or len(end_idx) == 0:
            continue

        start_idx = start_idx[-1]
        end_idx = end_idx[0]

        duration = (end_idx - start_idx + 1) / fs
        if not (duration_min <= duration <= duration_max):
            continue

        events.append((start_idx, end_idx))

    # merge
    merged = []
    for start_idx, end_idx in sorted(events):
        if not merged:
            merged.append([start_idx, end_idx])
        else:
            pre_start_idx, pre_end_idx = merged[-1]
            if (start_idx - pre_end_idx) / fs <= merge_interval:
                merged[-1][1] = end_idx
            else:
                merged.append([start_idx, end_idx])

    # remove too long
    merged = [
        (start, end) for start, end in merged if (end - start + 1) / fs <= duration_max
    ]

    # Final artifact check
    buffer_ms = 50  # 50 ms
    buffer = int(buffer_ms * fs / 1000)

    spindles = []

    for start_idx, end_idx in merged:
        # bounds safe
        start_buf = max(0, start_idx - buffer)

        end_buf = min(n_samples, end_idx + buffer)

        # spindle must be artifact free
        if not np.all(valid_mask[start_idx : end_idx + 1]):
            continue

        # buffer region must also be clean
        if not np.all(valid_mask[start_buf : end_buf + 1]):
            continue

        # peak detection
        peak_idx = start_idx + np.argmax(x[start_idx : end_idx + 1])

        # peak must be clean
        if not valid_mask[peak_idx]:
            continue

        spindles.append([start_idx, peak_idx, end_idx])

    spindles = np.asarray(spindles, dtype=int)

    if len(spindles) == 0:
        return np.empty((0, 7))

    # remove duplicate detections
    _, idx = np.unique(spindles, axis=0, return_index=True)
    spindles = spindles[np.sort(idx)]

    spindle_features = []

    for start_idx, peak_idx, end_idx in spindles:
        duration_spindle = (end_idx - start_idx + 1) / fs

        envelope_segment = envelope[start_idx : end_idx + 1]
        amplitude_spindle = np.max(envelope_segment)

        signal_segment = signal[start_idx : end_idx + 1]
        peak_frequency_spindle, mean_frequency_spindle = compute_peak_mean_frequency(
            signal_segment, fs, freq_band=freq_range
        )

        spindle_features.append(
            [
                start_idx,
                peak_idx,
                end_idx,
                duration_spindle,
                amplitude_spindle,
                peak_frequency_spindle,
                mean_frequency_spindle,
            ]
        )

    if len(spindle_features) == 0:
        return np.empty((0, 7))

    return np.asarray(spindle_features)
