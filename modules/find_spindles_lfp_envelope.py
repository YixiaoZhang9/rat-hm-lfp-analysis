import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import get_window, hilbert

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


def find_spindles_lfp_envelope(
    raw_signal,
    signal,
    fs,
    threshold=2.5,
    minpeak=5,
    durations=(0.4, 3.5),
    min_distance=0.5,
):
    """
    Detect thalamo-cortical sleep spindles from spindle-band filtered LFP
    using a Hilbert-envelope thresholding approach with artifact rejection.


    Parameters
    ----------
    raw_signal : array_like, shape (n_samples,)
        Raw LFP signal used for artifact detection.

    signal : array_like, shape (n_samples,)
        Spindle-band filtered LFP signal, typically filtered between
        9 and 20 Hz.

    fs : float
        Sampling frequency in Hz.

    threshold : float, optional
        Detection threshold applied to the smoothed z-scored Hilbert
        envelope. Candidate spindle events are initially defined as
        periods where the envelope exceeds this threshold.

        Default is 2.5.

    minpeak : float, optional
        Minimum required peak value of the smoothed z-scored envelope.
        Candidate events whose peak envelope is below this value are
        discarded.

        Default is 5.

    durations : tuple of float, optional
        Minimum and maximum allowed spindle duration in seconds, given as
        ``(min_duration, max_duration)``.

        Default is ``(0.4, 3.5)``.

    min_distance : float, optional
        Minimum temporal gap, in seconds, required to keep two candidate
        events separate. Events separated by less than this interval are
        merged.

        Default is 0.5 seconds.

    Returns
    -------
    spindles : ndarray, shape (n_events, 7)
        Detected spindle events and their features. Columns are:

        0. start index
        1. peak index
        2. end index
        3. duration in seconds
        4. peak Hilbert-envelope amplitude
        5. FFT-based peak frequency in Hz within the spindle band
        6. FFT-based power-weighted mean frequency in Hz within the
           spindle band

        If no spindles are detected, an empty array with shape ``(0, 7)``
        is returned.

    References
    ----------
    Zugaro, M. B., et al.
    FMAToolbox: FindSpindles.m.
    https://github.com/michael-zugaro/FMAToolbox

    """

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

    # Constants
    minBoutDuration = 0.050
    n = len(signal)
    time = np.arange(n) / fs

    # extract the envelope
    envelope = np.abs(hilbert(signal))
    # only use clean signal for normalization
    valid_env = envelope[valid_mask]

    mu = np.mean(valid_env)
    sigma = np.std(valid_env)

    zscored_envelope = (envelope - mu) / sigma

    # smooth the envelope
    sigma_sec = 0.04  # seconds
    # Characterization of Topographically Specific Sleep Spindles in Mice
    sigma_samples = sigma_sec * fs
    smoothed_envelope = gaussian_filter1d(zscored_envelope, sigma=sigma_samples)

    # Find start and stop as indices of threshold crossings
    above = (smoothed_envelope > threshold) & valid_mask
    above_int = above.astype(int)

    start_idx_spindle = np.where(np.diff(np.concatenate(([0], above_int))) == 1)[0]

    end_idx_spindle = np.where(np.diff(np.concatenate((above_int, [0]))) == -1)[0]

    if len(start_idx_spindle) == 0 or len(end_idx_spindle) == 0:
        return np.empty((0, 7))

    # discard incomplete pairs
    if end_idx_spindle[0] < start_idx_spindle[0]:
        end_idx_spindle = end_idx_spindle[1:]

    if len(start_idx_spindle) == 0 or len(end_idx_spindle) == 0:
        return np.empty((0, 7))

    if start_idx_spindle[-1] > end_idx_spindle[-1]:
        start_idx_spindle = start_idx_spindle[:-1]

    # discard short events
    durations_event = time[end_idx_spindle] - time[start_idx_spindle]
    idx_keep_events = durations_event >= minBoutDuration
    start_idx_spindle = start_idx_spindle[idx_keep_events]
    end_idx_spindle = end_idx_spindle[idx_keep_events]

    # Check if any events left
    if len(start_idx_spindle) == 0:
        return np.empty((0, 7))

    # Merge events when they are too close
    merged_start_idx = [start_idx_spindle[0]]
    merged_stop_idx = [end_idx_spindle[0]]

    for i in range(1, len(start_idx_spindle)):
        if time[start_idx_spindle[i]] - time[merged_stop_idx[-1]] < min_distance:
            merged_stop_idx[-1] = end_idx_spindle[i]
        else:
            merged_start_idx.append(start_idx_spindle[i])
            merged_stop_idx.append(end_idx_spindle[i])

    start_idx_spindle = np.array(merged_start_idx)
    end_idx_spindle = np.array(merged_stop_idx)

    # Discard events when envelope peak is too small
    peak_idx_spindle = []
    peak_val_spindle = []

    for start, end in zip(start_idx_spindle, end_idx_spindle):
        segment = smoothed_envelope[start:end]
        if len(segment) == 0:
            continue
        i = np.argmax(segment)
        peak_idx_spindle.append(start + i)
        peak_val_spindle.append(segment[i])

    peak_idx_spindle = np.array(peak_idx_spindle)
    peak_val_spindle = np.array(peak_val_spindle)

    # Peak threshold
    idx_keep = peak_val_spindle >= minpeak
    start_idx_spindle = start_idx_spindle[idx_keep]
    end_idx_spindle = end_idx_spindle[idx_keep]
    peak_idx_spindle = peak_idx_spindle[idx_keep]
    peak_val_spindle = peak_val_spindle[idx_keep]

    # Spindle start may be inaccurate due to leading delta wave, we need to correct for this
    # 1) Update threshold to 1/3 peak (if this is higher than previous threshold)
    # 2) Select spindles that need correction
    # 3) Find last threshold crossing before peak (vectorized code)
    new_threshold = peak_val_spindle / 3

    for i in range(len(start_idx_spindle)):
        if new_threshold[i] <= threshold:
            continue

        segment_idx = np.arange(start_idx_spindle[i], peak_idx_spindle[i])
        event_segment = smoothed_envelope[segment_idx]

        below = event_segment < new_threshold[i]

        if np.any(below):
            last_idx = segment_idx[np.where(below)[0][-1]]
            start_idx_spindle[i] = last_idx + 1

    # Duration filter
    duration = (end_idx_spindle - start_idx_spindle) / fs
    keep = (duration >= durations[0]) & (duration <= durations[1])

    start_idx_spindle = start_idx_spindle[keep]
    end_idx_spindle = end_idx_spindle[keep]
    peak_idx_spindle = peak_idx_spindle[keep]

    # Final artifact check
    buffer_ms = 50  # 50 ms
    buffer = int(buffer_ms * fs / 1000)

    clean_start = []
    clean_stop = []
    clean_peak = []

    n = len(signal)

    for start, end, peak in zip(start_idx_spindle, end_idx_spindle, peak_idx_spindle):
        # bounds safe
        s_buf = max(0, start - buffer)
        e_buf = min(n, end + buffer)

        # full spindle must be artifact free
        if not np.all(valid_mask[start:end]):
            continue

        # buffer region must also be clean
        if not np.all(valid_mask[s_buf:e_buf]):
            continue

        # peak must be clean
        if not valid_mask[peak]:
            continue

        clean_start.append(start)
        clean_stop.append(end)
        clean_peak.append(peak)

    start_idx_spindle = np.array(clean_start)
    end_idx_spindle = np.array(clean_stop)
    peak_idx_spindle = np.array(clean_peak)

    # calculate the features of spindles
    duration_spindle = []
    amplitude_spindle = []
    peak_frequency_spindle = []
    mean_frequency_spindle = []

    for start, end, peak in zip(start_idx_spindle, end_idx_spindle, peak_idx_spindle):
        # Duration / s
        duration = (end - start) / fs
        duration_spindle.append(duration)

        # Amplitude: raw Hilbert envelope peak
        envelope_segment = envelope[start : end + 1]
        amplitude = np.max(envelope_segment)
        amplitude_spindle.append(amplitude)

        # Frequency: filtered spindle segment
        signal_segment = signal[start : end + 1]
        peak_freq, mean_freq = compute_peak_mean_frequency(
            signal_segment, fs, freq_band=(9, 20)
        )

        peak_frequency_spindle.append(peak_freq)
        mean_frequency_spindle.append(mean_freq)

    # Output
    spindles = np.column_stack(
        (
            start_idx_spindle,
            peak_idx_spindle,
            end_idx_spindle,
            duration_spindle,
            amplitude_spindle,
            peak_frequency_spindle,
            mean_frequency_spindle,
        )
    )

    return spindles
