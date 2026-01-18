import numpy as np
from scipy.signal import iirnotch,butter, filtfilt, hilbert,convolve, get_window
from scipy.signal.windows import gaussian
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

def filter_lfp(lfp, fs,freq_range):
    # Bandpass filter
    b, a = butter(2, np.array(freq_range)/(fs/2), btype='band')
    return filtfilt(b, a, lfp)

def smooth_signal(signal, fs, sigma):
    """Smooth the signal using a Gaussian filter."""
    # Define the standard deviation(sigma)
    smoothing_sigma = sigma * fs
    # Create a Gaussian window with a standard deviation
    window_size = smoothing_sigma * 3 * 2
    # in a Gaussian distribution, about 99.7 % of the distribution's area lies within ±3σ.
    std = (window_size-1)/(2*sigma)
    gauss_filter = gaussian(window_size, std)
    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / sum(gauss_filter)
    # Apply the filter using convolution
    smoothed_lfp = convolve(signal, gauss_filter, 'same')
    return smoothed_lfp

def merge_close_ripples(start_indices, end_indices, fs, close_thresh):
    """ Merge ripples that are close to each other.
    parameters
    -starts : list of int Start indices of candidate ripples.
    -ends : list of int End indices of candidate ripples.
    -fs : float Sampling frequency (Hz).
    -close_thresh : float Minimum interval between ripples to avoid merging (seconds).

    Returns
    -merged_starts : list of int Start indices of merged ripples.
    -merged_ends : list of int End indices of merged ripples.
    """
    start_indices = np.array(start_indices)
    end_indices = np.array(end_indices)
    i = 0
    while i < len(start_indices) - 1:
        interval_duration = (start_indices[i + 1] - end_indices[i]) / fs
        if interval_duration <= close_thresh:
            # Merge the ripples
            start_indices[i + 1] = start_indices[i]
            start_indices = np.delete(start_indices, i)
            end_indices = np.delete(end_indices, i)
        else:
            i += 1
    return start_indices.tolist(), end_indices.tolist()



def fastrms(signal, window_size=5, dim=0, apply_amplitude_correction=False):
    """
    Calculate the instantaneous root-mean-square (RMS) of a signal.

    Parameters:
    - signal: 1D or 2D numpy array of the input signal.
    - window_size: Size of the moving window for RMS calculation.
    - dim: Dimension along which to compute the RMS (0 for rows, 1 for columns).
    - apply_amplitude_correction: If True, applies a correction for sinusoidal signals.

    Returns:
    - rms: The instantaneous RMS of the input signal.
    """
    if signal.ndim > 2:
        raise ValueError("Input signal must be 1D or 2D.")
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    # Create rectangular window
    window = get_window('boxcar', window_size)
    # Calculate power
    power = signal ** 2

    # Convolve to get RMS
    if signal.ndim == 1:
        rms = convolve(power, window, mode='same')
    else:  # 2D case
        rms = np.array([convolve(power[:, col], window, mode='same') for col in range(signal.shape[1])]).T

    # Normalize and compute RMS
    rms = np.sqrt(rms / np.sum(window))

    if apply_amplitude_correction:
        rms *= np.sqrt(2)

    return rms


def find_ripples_karlsson(lfp, fs, min_duration=0.015, zscore_thresh=3.0, smoothing_sigma=0.004,f_plot = 0,threshold=None):
    '''
    ----------------------------------------------------------------------------------------------
         reference:
     [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote experiences in the hippocampus.
         Nature Neuroscience 12, 913-918.
     [2] https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/detectors.py
    ---------------------------------------------------------------------------------------------
    minimum_duration : Minimum time the z-score has to stay above threshold to be considered a ripple.
                   The default is given assuming time is in units of seconds.
    zscore_threshold : Number of standard deviations the ripple power must exceed to be considered a ripple
    smoothing_sigma : Amount to smooth the time series over time. The default is given assuming time is in units of seconds.
    threshold: optional
        Envelop threhosld for ripple detection.if none, it will be set to `zscore_threshold`.

     '''
    filtered_lfp = filter_lfp(lfp, fs, [80, 250])
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, smoothing_sigma)

    zscored_envelope = zscore(smoothed_envelope)
    above_mean = zscored_envelope > 0
    if threshold is not None:
        zscore_thresh_new = (threshold - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    else:
        zscore_thresh_new = zscore_thresh
    above_thresh = zscored_envelope > zscore_thresh_new

    thresh_envelope = zscore_thresh_new * np.std(smoothed_envelope) + np.mean(smoothed_envelope)

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


    if f_plot:
        start_times = np.unique(start_idx_ripple) / fs
        end_times = np.unique(end_idx_ripple) / fs
        t = np.arange(len(lfp)) / fs

        fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

        # 1. original signal
        axs[0].plot(t, lfp, color="k", lw=0.5, label="Raw LFP")
        axs[0].plot(t, filtered_lfp, color="r", lw=0.5, label="Filtered (80-250Hz)")
        axs[0].legend()
        axs[0].set_ylabel("LFP (uV)")

        # 2. envelop
        axs[1].plot(t, smoothed_envelope, color="b", lw=1, label="Smoothed envelope")
        axs[1].axhline(thresh_envelope, color="r", ls="--", label="Threshold")
        for s, e in zip(start_times, end_times):
            axs[1].axvspan(s, e, color="orange", alpha=0.3)
        axs[1].legend()
        axs[1].set_ylabel("Envelope")

        # 3. Z-score
        axs[2].plot(t, zscored_envelope, color="g", lw=1, label="Z-scored envelope")
        axs[2].axhline(zscore_thresh, color="r", ls="--", label="Z-score thresh")
        axs[2].axhline(0, color="k", ls=":")
        for s, e in zip(start_times, end_times):
            axs[2].axvspan(s, e, color="orange", alpha=0.3)
        axs[2].legend()
        axs[2].set_ylabel("Z-score")
        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


    return {
        "filtered_lfp" :filtered_lfp,
        "StartIndex": np.unique(start_idx_ripple) / fs,
        "PeakIndex":np.unique(peak_idx_ripple) / fs,
        "EndIndex": np.unique(end_idx_ripple) / fs,
        "thresh_envelope": thresh_envelope
    }
