import numpy as np
from scipy.signal import iirnotch,butter, filtfilt, hilbert,convolve, get_window
from scipy.signal.windows import gaussian
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

def filter_lfp(lfp, fs,freq_range):
    # Bandpass filter
    b, a = butter(4, np.array(freq_range)/(fs/2), btype='band')
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


def find_ripples_karlsson(lfp, fs, min_duration=0.015, zscore_thresh=3, smoothing_sigma=0.004,f_plot = 0,threshold=None):
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
    filtered_lfp = filter_lfp(lfp, fs, [100, 250])
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
                segment = filtered_lfp[start_idx_mean[idx]:end_idx_mean[idx]]
                peak_local = np.argmax(segment)
                peak_idx_ripple.append(start_idx_mean[idx] + peak_local)

    start_idx_ripple = np.unique(start_idx_ripple)
    end_idx_ripple = np.unique(end_idx_ripple)
    peak_idx_ripple = np.unique(peak_idx_ripple)

    start_times = start_idx_ripple / fs
    end_times = end_idx_ripple / fs
    peak_times = peak_idx_ripple / fs

    if f_plot:
        t = np.arange(len(lfp)) / fs

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # ===============================
        axs[0].plot(t, lfp, color="k", lw=0.6, label="Raw LFP")
        axs[0].set_title("Raw LFP")
        axs[0].set_ylabel("LFP (uV)")
        axs[0].legend()

        # ===============================

        axs[1].plot(t, filtered_lfp, color="gray", lw=0.5, alpha=0.7, label="Filtered LFP (100–250 Hz)")
        axs[1].plot(t, smoothed_envelope, color="b", lw=1.2, label="Smoothed envelope")
        axs[1].axhline(thresh_envelope, color="r", ls="--", label="Envelope threshold")

        for s, e in zip(start_times, end_times):
            axs[1].axvspan(s, e, color="orange", alpha=0.25)
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

    ripple_envelopes = []
    for s, e in zip(start_idx_ripple, end_idx_ripple):
        ripple_envelopes.append(smoothed_envelope[s:e])

    return {
        "StartIndex": start_idx_ripple,
        "PeakIndex": peak_idx_ripple,
        "EndIndex": end_idx_ripple,
        "thresh_envelope": thresh_envelope,
        "ripple_envelop" : ripple_envelopes
    }


def find_ripples_karlsson_modified(lfp, fs, min_duration=0.015,smoothing_sigma=0.004,f_plot = 0):
    '''
    ----------------------------------------------------------------------------------------------
         reference:
     [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote experiences in the hippocampus.
         Nature Neuroscience 12, 913-918.
     [2] https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/detectors.py

     [3] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A., Sosa, M., ... & Frank, L. M. (2017).
     Distinct hippocampal-cortical memory representations for experiences associated with movement versus immobility.
     Elife, 6, e27621.
    ---------------------------------------------------------------------------------------------
    minimum_duration : Minimum time the z-score has to stay above threshold to be considered a ripple.
                   The default is given assuming time is in units of seconds.
    zscore_peak_threshold: Number of standard deviations the ripple peak power must exceed to be considered a ripple
    smoothing_sigma : Amount to smooth the time series over time. The default is given assuming time is in units of seconds.
    close_ripple_threshold : Exclude ripples that occur within `close_ripple_threshold` of a previously detected ripple.
    ---------------------------------------------------------------------------------------------
     '''
    filtered_lfp = filter_lfp(lfp, fs, [80, 250])
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, smoothing_sigma)

    zscored_envelope = zscore(smoothed_envelope)
    above_mean = zscored_envelope > 0
    thresh_envelope_z = estimate_noise_mirror_threshold(zscored_envelope, percentile=0.995,plot = 0)
    above_thresh = zscored_envelope > thresh_envelope_z

    # converting to original envelop
    mean_env = np.mean(smoothed_envelope)
    std_env = np.std(smoothed_envelope)
    thresh_envelope = thresh_envelope_z * std_env + mean_env

    start_idx_mean = np.where(np.diff(np.concatenate(([0], above_mean.astype(int)))) == 1)[0]
    end_idx_mean = np.where(np.diff(np.concatenate((above_mean.astype(int), [0]))) == -1)[0]
    start_idx_threshold = np.where(np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1)[0]
    end_idx_threshold = np.where(np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1)[0]

    # extend the threshold
    start_idx_ripple, end_idx_ripple = [], []

    for i, start in enumerate(start_idx_threshold):
        end = end_idx_threshold[i]

        # find segment which contains threshold crossing
        in_mean_idx = np.where((start_idx_mean <= start) & (end_idx_mean >= end))[0]

        if len(in_mean_idx) > 0:
            idx = in_mean_idx[0]
            if (end - start) / fs > min_duration:
                start_idx_ripple.append(start_idx_mean[idx])
                end_idx_ripple.append(end_idx_mean[idx])


    if f_plot:
        start_times = np.unique(start_idx_ripple) / fs
        end_times = np.unique(end_idx_ripple) / fs
        t = np.arange(len(lfp)) / fs

        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

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

    return {
        "StartIndex": np.unique(start_idx_ripple) / fs,
        "EndIndex": np.unique(end_idx_ripple) / fs,
        "thresh_envelope": thresh_envelope
    }

def estimate_noise_mirror_threshold(signal_envelope, bins=200, percentile=0.9999,plot = 0):
    """
    Estimate an adaptive threshold for SWR detection based on the mirrored noise distribution method.

    Parameters
    ----------
    signal_envelope : np.ndarray
        The smoothed envelope power of the filtered LFP signal.
    bins : int, optional
        Number of histogram bins for estimating the power distribution. Default is 200.
    percentile : float, optional
        Percentile (e.g., 0.9999 = 99.99th) used as the detection threshold cutoff. Default is 0.9999.

    Returns
    -------
    thresh_envelope : float
        Adaptive detection threshold for the envelope power, derived from the mirrored noise distribution.
    """

    # 1. Compute the histogram of the envelope power distribution
    hist, bin_edges = np.histogram(signal_envelope, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Find the mode (peak) of the histogram — assumed to represent pure noise
    mode_idx = np.argmax(hist)
    mode_val = bin_centers[mode_idx]

    # 3. Take the left (low-power) side as the noise component
    left_side = bin_centers[bin_centers < mode_val]
    left_hist = hist[bin_centers < mode_val]

    # 4. Mirror the left-side distribution around the mode to approximate symmetry of the noise
    mirrored_bins = 2 * mode_val - left_side
    mirrored_bins = mirrored_bins[::-1]
    mirrored_hist = left_hist[::-1]  # reverse order for mirroring

    # 5. Combine left side + mode + mirrored right side
    noise_bins = np.concatenate([left_side, [mode_val], mirrored_bins])
    noise_pdf = np.concatenate([left_hist, [hist[mode_idx]], mirrored_hist])

    # Normalize PDF so it sums to 1
    noise_pdf /= np.sum(noise_pdf)

    # 6. Compute CDF
    cdf = np.cumsum(noise_pdf)
    cdf /= cdf[-1]

    # 7. Interpolate to find the envelope power at the desired percentile
    thresh_envelope = np.interp(percentile, cdf, noise_bins)

    # 8. Optional visualization
    if plot:
        sns.histplot(signal_envelope, bins=bins, kde=False, color='gray', label='Original distribution')
        plt.plot(noise_bins, noise_pdf * max(hist) / max(noise_pdf), 'r-', label='Mirrored noise PDF (scaled)')
        plt.axvline(mode_val, color='b', linestyle='--', label='Mode')
        plt.axvline(thresh_envelope, color='r', linestyle=':', label=f'{percentile * 100:.3f}th percentile')
        plt.legend()
        plt.xlabel('Envelope')
        plt.ylabel('Density')
        plt.title('Mirrored noise distribution and adaptive threshold')
        plt.show()

    return thresh_envelope




def find_ripples_dahal(lfp, fs, min_duration=0.02, zscore_thresh=2.0, zscore_peak_thresh=5.0, smoothing_sigma=0.004,
                       close_ripple_thresh=0.03):
    '''
    ---------------------------------------------------------------------------------------------------
    reference:
    [1] Dahal, P., Rauhala, O. J., Khodagholy, D., & Gelinas, J. N. (2023).Hippocampal–cortical coupling
        differentiates long-term memory processes. Proceedings of the National Academy of Sciences
    [2] https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/detectors.py
    ---------------------------------------------------------------------------------------------------
    minimum_duration : Minimum time the z-score has to stay above threshold to be considered a ripple.
                      The default is given assuming time is in units of seconds.
    zscore_threshold : Number of standard deviations the ripple power must exceed to be considered a ripple
    zscore_peak_threshold: Number of standard deviations the ripple peak power must exceed to be considered a ripple
    smoothing_sigma : Amount to smooth the time series over time. The default is given assuming time is in units of seconds.
    close_ripple_threshold : Exclude ripples that occur within `close_ripple_threshold` of a previously detected ripple.
    '''
    filtered_lfp = filter_lfp(lfp, fs, [80, 250])
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, smoothing_sigma)

    zscored_envelope = zscore(smoothed_envelope)
    above_thresh = zscored_envelope > zscore_thresh

    start_idx_threshold = np.where(np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1)[0]
    end_idx_threshold = np.where(np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1)[0]

    start_idx_ripple, end_idx_ripple = [], []

    for start in start_idx_threshold:
        end = end_idx_threshold[start_idx_threshold == start][0]
        peak_ripple = np.max(zscored_envelope[start:end])
        duration_ripple = (end - start) / fs

        if peak_ripple > zscore_peak_thresh and duration_ripple > min_duration:
            start_idx_ripple.append(start)
            end_idx_ripple.append(end)

    # Merge close ripples
    merged_starts,merged_ends = merge_close_ripples(start_idx_ripple, end_idx_ripple, fs, close_ripple_thresh)

    return {
        "StartIndex": np.array (merged_starts) / fs,
        "EndIndex": np.array(merged_ends) / fs
    }


def find_ripples_diba(lfp, fs, min_duration=0, zscore_thresh=2):
    '''
    ---------------------------------------------------------------------------------------------------
    reference:
    [1] Diba, K., & Buzsáki, G. (2007). Forward and reverse hippocampal place-cell sequences during ripples.
        Nature neuroscience.
    [2] Csicsvari, J., Hirase, H., Czurko, A., Mamiya, A., & Buzsáki, G. (1999). Fast network oscillations
        in the hippocampal CA1 region of the behaving rat. The Journal of neuroscience, 19(16), RC20.
    ---------------------------------------------------------------------------------------------------
    minimum_duration : Minimum time the z-score has to stay above threshold to be considered a ripple.
                      The default is given assuming time is in units of seconds.
    zscore_threshold : Number of standard deviations the ripple power must exceed to be considered a ripple
    zscore_peak_threshold: Number of standard deviations the ripple peak power must exceed to be considered a ripple
    smoothing_sigma : Amount to smooth the time series over time. The default is given assuming time is in units of seconds.
    close_ripple_threshold : Exclude ripples that occur within `close_ripple_threshold` of a previously detected ripple.
    '''
    filtered_lfp = filter_lfp(lfp, fs, [80, 250])
    window_size = int(0.005 * fs) # 5ms window
    power = fastrms(filtered_lfp, window_size)

    mean_power = np.mean(power)
    std_power = np.std(power)

    above_thresh = power > (mean_power + zscore_thresh * std_power)
    start_idx_threshold = np.where(np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1)[0]
    end_idx_threshold = np.where(np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1)[0]
    above_thresh2 = power > (mean_power + 1.5 * std_power)
    start_idx_threshold2 = np.where(np.diff(np.concatenate(([0], above_thresh2.astype(int)))) == 1)[0]
    end_idx_threshold2 = np.where(np.diff(np.concatenate((above_thresh2.astype(int), [0]))) == -1)[0]

    # extend the threshold to broader segment
    start_idx_ripple, end_idx_ripple = [], []

    for start, end in zip(start_idx_threshold, end_idx_threshold):
        in_mean_idx = np.where((start_idx_threshold2 <= start) & (end_idx_threshold2 >= end))[0]

        if len(in_mean_idx) > 0:
            idx = in_mean_idx[0]
            ripple_start = start_idx_threshold2[idx]
            ripple_end = end_idx_threshold2[idx]

            if (ripple_end - ripple_start) / fs >= min_duration:
                start_idx_ripple.append(ripple_start)
                end_idx_ripple.append(ripple_end)

    return {
        "StartIndex": np.unique(start_idx_ripple) / fs,
        "EndIndex": np.unique(end_idx_ripple) / fs
    }


def find_ripples_vicente(lfp, fs, min_duration=0.015, zscore_thresh=5, close_ripple_threshold=0.015):
    '''
    ---------------------------------------------------------------------------------------------------
    reference:
    [1] Vicente, A. F., Slézia, A., Ghestem, A., Bernard, C., & Quilichini, P. P. (2020).
        In vivo characterization of neurophysiological diversity in the lateral supramammillary
        nucleus during hippocampal sharp-wave ripples of adult rats. Neuroscience, _435_, 95-111.
    ---------------------------------------------------------------------------------------------------
    minimum_duration : Minimum time the z-score has to stay above threshold to be considered a ripple.
                      The default is given assuming time is in units of seconds.
    zscore_threshold : Number of standard deviations the ripple power must exceed to be considered a ripple
    zscore_peak_threshold: Number of standard deviations the ripple peak power must exceed to be considered a ripple
    smoothing_sigma : Amount to smooth the time series over time. The default is given assuming time is in units of seconds.
    close_ripple_threshold : Exclude ripples that occur within `close_ripple_threshold` of a previously detected ripple.
    '''
    filtered_lfp = filter_lfp(lfp, fs, [80, 250])
    window_size = int(0.005 * fs)
    power = fastrms(filtered_lfp, window_size)

    mean_power = np.mean(power)
    std_power = np.std(power)

    above_thresh = power > (mean_power + zscore_thresh * std_power)
    above_thresh2 = power > (mean_power + 0.5 * std_power)

    start_idx_threshold = np.where(np.diff(np.concatenate(([0], above_thresh.astype(int)))) == 1)[0]
    end_idx_threshold = np.where(np.diff(np.concatenate((above_thresh.astype(int), [0]))) == -1)[0]

    start_idx_threshold2 = np.where(np.diff(np.concatenate(([0], above_thresh2.astype(int)))) == 1)[0]
    end_idx_threshold2 = np.where(np.diff(np.concatenate((above_thresh2.astype(int), [0]))) == -1)[0]

    # extend the threshold using broader threshold segments
    start_idx_ripple, end_idx_ripple = [], []

    for start, end in zip(start_idx_threshold, end_idx_threshold):
        in_mean_idx = np.where((start_idx_threshold2 <= start) & (end_idx_threshold2 >= end))[0]

        if len(in_mean_idx) > 0:
            idx = in_mean_idx[0]
            ripple_start = start_idx_threshold2[idx]
            ripple_end = end_idx_threshold2[idx]

            if (ripple_end - ripple_start) / fs >= min_duration:
                start_idx_ripple.append(ripple_start)
                end_idx_ripple.append(ripple_end)

    # merge close ripples
    merged_starts, merged_ends = merge_close_ripples(start_idx_ripple, end_idx_ripple, fs, close_ripple_threshold)
    # Merge close ripples
    return {
        "StartIndex": np.array(merged_starts) / fs,
        "EndIndex": np.array(merged_ends) / fs
    }


def find_bouts(scoring_data, target_value=3, fs=1000):
    """
    Find all continuous segments (bouts) in scoring_data where the value equals target_value.

    Parameters
    ----------
    scoring_data : array-like
        1D array of sleep scoring values (e.g., 1=Wake, 3=NonREM, 5=REM, 4=intermediate)
    target_value : int or float, optional
        The state value to detect bouts for NREM sleep (default = 3)
    fs : float, optional
        Sampling frequency in Hz (default = 1000)

    Returns
    -------
    bouts : list of tuples
        List of (start_idx, end_idx, start_time, end_time), where:
            start_idx / end_idx are sample indices
            start_time / end_time are times in seconds
    """
    scoring_data = np.array(scoring_data)
    is_target = (scoring_data == target_value).astype(int)
    diff = np.diff(is_target, prepend=0, append=0)

    # 1 → bout starts, -1 → bout ends
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = [(s*fs, e*fs) for s, e in zip(starts, ends)]
    return bouts
