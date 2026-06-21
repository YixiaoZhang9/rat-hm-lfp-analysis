import numpy as np
import pywt
from scipy.ndimage import convolve1d
from scipy.signal import (
    detrend,
    filtfilt,
    find_peaks,
    firwin,
    get_window,
    hilbert,
    resample,
    resample_poly,
    welch,
)

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


def spindle_detection_wavelet(raw_signal, signal, fs_raw, thrL=1, thrH=3):
    """


    Detect sleep spindles using a continuous wavelet transform (CWT)-based
    spindle-band power detector with artifact rejection.

    This function implements a single-channel spindle detection approach
    adapted from Bandarabadi et al. (2020). The signal is detrended and
    downsampled to 200 Hz for wavelet-based detection. Spindle-band power is
    estimated using a CWT in the 10–14 Hz range, followed by 1/f correction,
    temporal smoothing, dual-threshold detection, duration filtering, cycle
    count validation, spectral specificity checking, and artifact rejection.

    Parameters
    ----------
    raw_signal : ndarray, shape (n_samples,)
        Raw LFP signal used for artifact detection.

    signal : ndarray, shape (n_samples,)
        Single-channel LFP signal used for spindle detection. This signal may
        be broadband or broadly filtered, but should contain the spindle
        frequency range.

    fs_raw : float
        Original sampling frequency in Hz.

    thrL : float, optional
        Lower threshold multiplier used for refining spindle boundaries.

        Default is 1.

    thrH : float, optional
        Upper threshold multiplier used for detecting spindle candidates.

        Default is 3.

    Returns
    -------
    spindles : ndarray, shape (n_events, 7)
        Detected spindle events and features. Columns are:

        0. start index in the original sampling rate
        1. peak index in the original sampling rate
        2. end index in the original sampling rate
        3. duration in seconds
        4. peak Hilbert-envelope amplitude
        5. FFT-based peak frequency in Hz within 9–20 Hz
        6. FFT-based power-weighted mean frequency in Hz within 9–20 Hz

        If no spindle is detected, returns an empty array with shape
        ``(0, 7)``.

    References:
    Bandarabadi, M., Herrera, C. G., Gent, T. C., Bassetti, C., Schindler, K., & Adamantidis, A. R. (2020).
    A role for spindles in the onset of rapid eye movement sleep. Nature communications, 11(1), 5247.

    """

    # Detect LFP artifacts using the median absolute deviation method
    (valid_times, artifact_intervals_s) = mad_artifact_detector(
        raw_signal,
        mad_thresh=6.0,
        proportion_above_thresh=0.1,
        removal_window_ms=100.0,
        sampling_frequency=fs_raw,
    )
    t = np.arange(len(signal)) / fs_raw

    filtered_signal_raw = np.asarray(signal).flatten()
    envelope_raw = np.abs(hilbert(filtered_signal_raw))

    # set parameters
    param = {
        "freq": [9, 10, 14, 16, 20],
        "thrF": [thrL, thrH, 20],  # original parameter "thrF": [1, 3, 20],
        "minDur": 0.4,
        "maxDur": 3.5,
        "minCycle": 5,
        "maxCycle": 30,
    }

    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input must be 1D single-channel signal.")

    # # Preprocessing
    signal = detrend(filtered_signal_raw)
    # Anti-alias + downsample
    target_fs = 200
    signal = resample_poly(signal, target_fs, fs_raw)
    fs = target_fs

    # Resample valid mask
    t_raw = np.arange(len(raw_signal)) / fs_raw

    valid_mask_raw = intervals_to_mask(valid_times, t_raw)

    valid_mask = resample(valid_mask_raw.astype(float), len(signal)) > 0.99

    # extract CWT coefficients in the spindle range
    wavelet = "fbsp2-1-2"
    freq_cent = np.arange(param["freq"][1], param["freq"][2] + 0.5, 0.5)
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * fs / freq_cent
    coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    # Power
    cwt_power = np.abs(coef) ** 2
    # 1/f correction (FIXED normalization)
    cwt_power = np.sum(freq_cent[:, None] * cwt_power, axis=0)

    # Smoothing (200 ms Hanning)
    win_len = max(3, int(fs / 5))
    window = np.hanning(win_len)
    window /= window.sum()
    x = convolve1d(cwt_power, window, mode="nearest")

    # filtering in the spindle range to count number of cycles
    order = int(round(3 * (fs / param["freq"][0])))
    taps = firwin(
        order + 1, [param["freq"][0], param["freq"][4]], fs=fs, pass_zero=False
    )
    if len(signal) <= 3 * (len(taps) - 1):
        return np.empty((0, 7))
    filt_data = filtfilt(taps, [1], signal)

    # find upper and lower thresholds
    # removing outliers for a proper estimation of std-based thresholds
    valid_x = x[valid_mask]

    if len(valid_x) == 0:
        return np.empty((0, 7))

    # remove extreme outliers
    valid_x = valid_x[valid_x < 10 * np.nanstd(valid_x)]

    if len(valid_x) == 0:
        return np.empty((0, 7))

    mu = np.nanmean(valid_x)
    sigma = np.nanstd(valid_x)
    if sigma <= 0:
        return np.empty((0, 7))
    # lower threshold
    thrL = mu + param["thrF"][0] * sigma
    # upper threshold
    thrH = mu + param["thrF"][1] * sigma
    # outlier threshold
    thrM = mu + param["thrF"][2] * sigma

    # Candidate detection
    diff = np.diff(np.sign(x - thrH))
    # detection using the upper thr, finding start/end using the lower thr
    starts = np.where(diff == 2)[0] + 1
    ends = np.where(diff == -2)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return np.empty((0, 7))

    # align
    if ends[0] < starts[0]:
        ends = ends[1:]

    if len(starts) > len(ends):
        starts = starts[:-1]

    refined_s, refined_e = [], []

    for start, end in zip(starts, ends):
        left = np.where((x[:start] - thrL) < 0)[0]
        right = np.where((x[end:] - thrL) < 0)[0]

        if len(left) == 0 or len(right) == 0:
            continue

        refined_s.append(left[-1])
        refined_e.append(end + right[0])

    refined_s = np.array(refined_s)
    refined_e = np.array(refined_e)

    # go through each detected event and check spindle conditions
    events = []

    for start, end in zip(refined_s, refined_e):
        # check min/max duration
        duration = (end - start) / fs
        if not (param["minDur"] <= duration <= param["maxDur"]):
            continue

        # check min/max number of cycles
        segment = filt_data[start : end + 1]
        peaks, _ = find_peaks(segment)
        if not (param["minCycle"] <= len(peaks) <= param["maxCycle"]):
            continue

        # check maximum power for outlier removal
        if np.max(cwt_power[start : end + 1]) > thrM:
            continue

        # check if power increase is spindle specific
        seg2 = signal[max(0, start - fs // 4) : min(len(signal), end + fs // 4)]
        nfft_val = max(len(seg2), 2 * fs)
        f, pxx = welch(
            seg2,
            fs=fs,
            window="hann",
            nperseg=len(seg2),
            nfft=nfft_val,
            noverlap=0,
            detrend=False,
        )

        pxx = f * pxx  # 1/f correction
        band = (f >= param["freq"][0]) & (f <= param["freq"][4])
        pxx_spin = pxx[band]

        mask_low = (f >= 6) & (f <= 8.5)
        mask_high = (f >= 22) & (f <= 30)
        noise_mask = mask_low | mask_high

        if np.any(noise_mask):
            pxx_noise = np.percentile(pxx[noise_mask], 95)
        else:
            pxx_noise = 0
        if len(pxx_spin) == 0 or np.max(pxx_spin) < pxx_noise:
            continue
        events.append((start, end))

    # Final artifact rejection
    buffer_ms = 50
    buffer = int(buffer_ms * fs / 1000)

    clean_events = []
    for start, end in events:
        s_buf = max(0, start - buffer)

        e_buf = min(len(valid_mask) - 1, end + buffer)

        # event must be clean
        if not np.all(valid_mask[start : end + 1]):
            continue

        # buffer must be clean
        if not np.all(valid_mask[s_buf : e_buf + 1]):
            continue

        peak = start + np.argmax(x[start : end + 1])

        # peak must be clean
        if not valid_mask[peak]:
            continue

        clean_events.append([start, peak, end])

    # scale the final detection results because we resmapled before
    scale = fs_raw / fs

    spindles = np.array(clean_events)

    if len(spindles) > 0:
        spindles = (spindles * scale).astype(int)
        # remove duplicate detections after rescaling
        _, idx = np.unique(spindles, axis=0, return_index=True)
        idx = np.sort(idx)
        spindles = spindles[idx]
    else:
        return np.empty((0, 7))

    spindle_features = []

    for start_idx, peak_idx, end_idx in spindles:
        duration_spindle = (end_idx - start_idx + 1) / fs_raw

        envelope_segment = envelope_raw[start_idx : end_idx + 1]
        amplitude_spindle = np.max(envelope_segment)

        signal_segment = filtered_signal_raw[start_idx : end_idx + 1]
        peak_frequency_spindle, mean_frequency_spindle = compute_peak_mean_frequency(
            signal_segment, fs_raw, freq_band=[9, 20]
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
