import numpy as np
from scipy.signal import iirnotch,butter, filtfilt, hilbert,convolve, get_window
from scipy.signal.windows import gaussian
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from modules.lfp_artifact_MAD_detection import mad_artifact_detector

def filter_lfp(lfp, fs,freq_range):
    # Bandpass filter
    b, a = butter(4, np.array(freq_range)/(fs/2), btype='band')
    return filtfilt(b, a, lfp)

def smooth_signal(signal, fs, sigma):
    """Smooth the signal using a Gaussian filter."""
    # Define the standard deviation(sigma)
    smoothing_sigma = sigma * fs
    # Create a Gaussian window with a standard deviation
    window_size = int(smoothing_sigma * 3 * 2)
    if window_size % 2 == 0:
        window_size += 1
    # in a Gaussian distribution, about 99.7 % of the distribution's area lies within ±3σ.
    # std = (window_size - 1) / (2 * sigma)
    # gauss_filter = gaussian(window_size, std)

    gauss_filter = gaussian(window_size, smoothing_sigma)
    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / sum(gauss_filter)
    # Apply the filter using convolution
    smoothed_lfp = convolve(signal, gauss_filter, 'same')
    return smoothed_lfp


def intervals_to_mask(intervals, timestamps):
    mask = np.zeros_like(timestamps, dtype=bool)
    for start, end in intervals:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask

def compute_peak_mean_frequency(segment, fs, window_type='hamming',freq_band=(100, 250)):
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


def find_ripples_karlsson_Adaptive(lfp, fs, min_duration=0.015, zscore_thresh=3, smoothing_sigma=0.006,f_plot = 0):
    """
    Detect hippocampal ripples using an adaptive Karlsson & Frank (2009)-based ripple detection algorithm with
    automatic artifact rejection.

    Parameters
    ----------
    lfp : ndarray, shape (n_samples,)
        Raw local field potential (LFP) signal.

    fs : float
        Sampling frequency in Hz.

    min_duration : float, optional
        Minimum duration (seconds) that the ripple envelope must remain
        above the high threshold (`zscore_thresh`) to be considered
        a valid ripple.

        Default = 0.015 s (15 ms).

    zscore_thresh : float, optional
        High detection threshold applied to the z-scored ripple envelope.

        Default = 3.

    smoothing_sigma : float, optional
        Standard deviation (seconds) of the Gaussian kernel used to
        smooth the Hilbert envelope.

        Default = 0.006 s (6 ms).

    f_plot : bool, optional
        If True, generate diagnostic plots showing:

        - Raw LFP
        - Ripple-band filtered LFP
        - Smoothed envelope
        - Detected ripples
        - Artifact regions

        Default = False.

    Returns
    -------
    dict
        Dictionary containing ripple events and extracted features:

        - "StartIndex" : ndarray
            Ripple start sample indices.

        - "PeakIndex" : ndarray
            Ripple peak sample indices.

        - "EndIndex" : ndarray
            Ripple end sample indices.

        - "Duration" : list of float
            Ripple durations in seconds.

        - "Amplitude" : list of float
            Peak Hilbert-envelope amplitudes.

        - "Peak_Frequency" : list of float
            Dominant frequency (Hz) estimated from each ripple event.

        - "Mean_Frequency" : list of float
            Mean spectral frequency (Hz) of each ripple event.

    Notes
    -----
    Differences from the original Karlsson detector:

    1. Artifact rejection using a MAD-based detector is applied before
       ripple detection.

    2. Envelope normalization is computed using only artifact-free
       periods.

    3. Ripple boundaries are expanded using a low threshold of
       z > 0.5 rather than z > 0.

    4. Candidate ripples are rejected if artifacts occur within
       ±50 ms of the event.

    References
    ----------
    Karlsson, M. P., & Frank, L. M. (2009).
    Awake replay of remote experiences in the hippocampus.
    Nature Neuroscience, 12(7), 913–918.

    https://github.com/Eden-Kramer-Lab/ripple_detection
    """

    # Detect LFP artifacts using the median absolute deviation method
    (valid_times, artifact_intervals_s) = mad_artifact_detector(lfp, mad_thresh = 6.0, proportion_above_thresh = 0.1,
    removal_window_ms = 100.0,sampling_frequency = fs)

    # convert intervals to mask
    t = np.arange(len(lfp)) / fs
    valid_mask = intervals_to_mask(valid_times, t)

    filtered_lfp = filter_lfp(lfp, fs, [100, 250])
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, smoothing_sigma)

    valid_env = smoothed_envelope[valid_mask]
    mu = np.mean(valid_env)
    sigma = np.std(valid_env)
    zscored_envelope = (smoothed_envelope - mu) / sigma

    above_mean = (zscored_envelope > 0.5) & valid_mask
    above_thresh = (zscored_envelope > zscore_thresh) & valid_mask

    start_idx_mean = np.where(np.diff(np.concatenate(([0], above_mean.astype(int)))) == 1)[0]
    end_idx_mean = np.where(np.diff(np.concatenate((above_mean.astype(int), [0]))) == -1)[0]
    start_idx_threshold = np.where(np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1)[0]
    end_idx_threshold = np.where(np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1)[0]

    # extend the threshold
    start_idx_ripple, peak_idx_ripple, end_idx_ripple = [], [], []

    for i, start in enumerate(start_idx_threshold):
        end = end_idx_threshold[i]

        # find segment which contains threshold crossing
        in_mean_idx = np.where((start_idx_mean <= start) & (end_idx_mean >= end))[0]

        if len(in_mean_idx) > 0:
            idx = in_mean_idx[0]
            if (end - start) / fs > min_duration:
                start_idx_ripple.append(start_idx_mean[idx])
                end_idx_ripple.append(end_idx_mean[idx])
                # find the peak
                segment = smoothed_envelope[start_idx_mean[idx]:end_idx_mean[idx]]
                peak_local = np.argmax(segment)
                peak_idx_ripple.append(start_idx_mean[idx] + peak_local)

    # final artifact mask check
    buffer_ms = 50  # 10–50 ms
    buffer = int(buffer_ms * fs / 1000)
    clean_start = []
    clean_end = []
    clean_peak = []

    for start, end, peak in zip(start_idx_ripple, end_idx_ripple, peak_idx_ripple):

        # full ripple must be clean
        if not np.all(valid_mask[start:end]):
            continue

        # buffer zone check
        s0 = max(0, start - buffer)
        e0 = min(len(valid_mask), end + buffer)

        if not np.all(valid_mask[s0:e0]):
            continue

        clean_start.append(start)
        clean_end.append(end)
        clean_peak.append(peak)

    start_idx_ripple = np.array(clean_start)
    end_idx_ripple = np.array(clean_end)
    peak_idx_ripple = np.array(clean_peak)


   #calculate the features of ripples
    duration_ripple = []
    amplitude_ripple = []
    peak_frequency_ripple = []
    mean_frequency_ripple = []

    for start, end, peak in zip(start_idx_ripple, end_idx_ripple, peak_idx_ripple):
        # Duration / s
        duration = (end - start) / fs
        duration_ripple.append(duration)

        # Amplitude: raw Hilbert envelope peak
        envelope_segment = instantaneous_amplitude[start:end + 1]
        amplitude = np.max(envelope_segment)
        amplitude_ripple.append(amplitude)

        # Frequency: filtered ripple segment
        signal_segment = filtered_lfp[start:end + 1]
        peak_freq, mean_freq = compute_peak_mean_frequency(signal_segment, fs)

        peak_frequency_ripple.append(peak_freq)
        mean_frequency_ripple.append(mean_freq)


    start_times = start_idx_ripple / fs
    end_times = end_idx_ripple / fs
    peak_times = peak_idx_ripple / fs

    if f_plot:
        artifact_mask = ~valid_mask
        artifact_start = np.where(np.diff(np.concatenate(([0], artifact_mask.astype(int)))) == 1)[0]
        artifact_end = np.where(np.diff(np.concatenate((artifact_mask.astype(int), [0]))) == -1)[0]
        t = np.arange(len(lfp)) / fs

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        axs[0].plot(t, lfp, color="k", lw=0.6, label="Raw LFP")
        axs[0].set_title("Raw LFP")
        axs[0].set_ylabel("LFP (uV)")
        axs[0].legend()

        axs[1].plot(t, filtered_lfp, color="gray", lw=0.5, alpha=0.7, label="Filtered LFP (100–250 Hz)")
        axs[1].plot(t, smoothed_envelope, color="b", lw=1.2, label="Smoothed envelope")
        # ripple spans
        for start, end in zip(start_times, end_times):
            axs[1].axvspan(start, end, color="orange", alpha=0.25)

        # artifact spans
        for i, (start, end) in enumerate(zip(artifact_start, artifact_end)):
            axs[1].axvspan(start / fs, end / fs, color="red", alpha=0.2,
                           label="Artifact" if i == 0 else None)

        axs[1].scatter(peak_times, smoothed_envelope[peak_idx_ripple],
                       color="magenta", s=30, label="Ripple peaks")

        axs[1].scatter(start_times, smoothed_envelope[start_idx_ripple],
                       color="green", s=25, label="Ripple start points")

        axs[1].scatter(end_times, smoothed_envelope[end_idx_ripple],
                       color="pink", s=25, label="Ripple end points")

        axs[1].set_title("Filtered LFP + Envelope + Ripple Detection")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        # plt.savefig('test.png')

    # ripple_envelopes = []
    # for start, end in zip(start_idx_ripple, end_idx_ripple):
    #     ripple_envelopes.append(smoothed_envelope[start:end])

    return {
        "StartIndex": start_idx_ripple,
        "PeakIndex": peak_idx_ripple,
        "EndIndex": end_idx_ripple,
        "Duration":duration_ripple,
        "Amplitude" :amplitude_ripple,
        "Peak_Frequency":peak_frequency_ripple,
        "Mean_Frequency": mean_frequency_ripple
    }



