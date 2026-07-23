# this file is the fuction used in preprocessing the data

import os
import re

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import butter, filtfilt, freqz, hilbert, resample,resample_poly
from fractions import Fraction


matplotlib.use("Qt5Agg")
import glob
import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pywt
from PyQt5.QtWidgets import QApplication

import modules.powerline_noise_removal as powerline
from modules.ephys_signal_view import SignalPlotViewer


def bandpass_filter(data, lowcut=80, highcut=250, fs=1000, order=4):

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def downsampling(data, fs, ds_fs, plot_response=False):
    """
    Downsample the input data from original sampling rate fs to new sampling rate ds_fs.

    Parameters:
        data (np.ndarray): Input signal data. Can be 1D or 2D (channels x time).
        fs (float): Original sampling frequency (Hz).
        ds_fs (float): Desired downsampled frequency (Hz).
        plot_response (bool): If True, plot the frequency and phase response of the filter.

    Returns:
        np.ndarray: Downsampled signal.
    """
    if ds_fs >= fs:
        raise ValueError(
            "Downsample frequency must be less than original sampling frequency."
        )

    # Design low-pass filter (Butterworth)
    nyq = fs / 2
    cutoff = ds_fs / 2  # anti-aliasing
    b, a = butter(N=4, Wn=cutoff / nyq, btype="low")

    # Optional: plot frequency & phase response
    if plot_response:
        w, h = freqz(b, a, worN=8000)
        freqs = w * fs / (2 * np.pi)

        plt.figure(figsize=(12, 5))

        # Magnitude response
        plt.subplot(1, 2, 1)
        plt.plot(freqs, 20 * np.log10(abs(h)), "b")
        plt.axvline(cutoff, color="r", linestyle="--", label=f"Cutoff = {cutoff} Hz")
        plt.title("Magnitude Response (dB)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()

        # Phase response
        plt.subplot(1, 2, 2)
        angles = np.unwrap(np.angle(h))
        plt.plot(freqs, angles, "g")
        plt.title("Phase Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Apply filter
    if data.ndim == 1:
        filtered = filtfilt(b, a, data)
    else:
        filtered = np.vstack([filtfilt(b, a, ch) for ch in data])

    # ratio = Fraction(ds_fs / fs).limit_denominator()
    #
    # if data.ndim == 1:
    #     downsampled = resample_poly(
    #         filtered,
    #         up=ratio.numerator,
    #         down=ratio.denominator
    #     ).astype("float32", copy=False)
    # Calculate number of samples after resampling
    num_samples = int(filtered.shape[-1] * ds_fs / fs)

    # Resample
    if data.ndim == 1:
        downsampled = resample(filtered, num_samples)
    else:
        downsampled = np.vstack([resample(ch, num_samples) for ch in filtered])

    return downsampled


def powerline_filter(data, fs, powerline_freq, Method="Notch", plot_data=False):
    """
    Apply a notch filter to remove powerline noise at a given frequency.

    Parameters:
        data (np.ndarray): Input signal data. Can be 1D or 2D (channels x time).
        powerline_freq (float): Frequency to be removed (e.g., 50 or 60 Hz).
        fs (float): Sampling frequency of the data.
        plot_data (bool): If True, plot the original data and filterd data.
        Method (str): Method to apply to filtered data.


        plot_response (bool): If True, plot the frequency and phase response of the filter.

    Returns:
        np.ndarray: Filtered signal with powerline interference removed.
    """

    if Method == "Notch":
        filt_data = powerline.ft_preproc_notch(
            data, fs, powerline_freq, plot_response=False
        )

    elif Method == "DFT":
        dftbandwidth = [1] * len(np.atleast_1d(powerline_freq))
        dftneighbourwidth = [3] * len(np.atleast_1d(powerline_freq))
        # filt_data = powerline.ft_preproc_dftfilter(data, fs, powerline_freq, 'zero')
        filt_data = powerline.ft_preproc_dftfilter(
            data,
            fs,
            powerline_freq,
            "neighbour",
            dftbandwidth=dftbandwidth,
            dftneighbourwidth=dftneighbourwidth,
        )

    elif Method == "Adaptive_LMS":
        filt_data = data  # Start with the original data
        for harmonic_freq in powerline_freq:
            filt_data = powerline.ft_preproc_adaptivefilter(
                filt_data,
                fs,
                powerline_freq=harmonic_freq,  # Wrap in list if needed
            )

    elif Method == "Adaptive_RLS":
        filt_data = data  # Start with the original data
        for harmonic_freq in powerline_freq:
            filt_data = powerline.ft_preproc_adaptivefilter_rls(
                filt_data, fs, powerline_freq=harmonic_freq, lambda_=0.998, delta=0.5
            )

    # Optional: plot raw data and filterd data
    if plot_data:
        app_created = False
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            app_created = True

        squeezed_data = data.squeeze()
        squeezed_filt_data = filt_data.squeeze()

        # create dic
        data_dict = {"Original": squeezed_data, "Filtered": squeezed_filt_data}

        window = SignalPlotViewer(data_dict, fs, window_sec=5)
        window.show()

        if app_created:
            app.exec()

        plot_fft(data, fs, title="Original Signal FFT")
        plot_fft(filt_data, fs, title="Filtered Signal FFT")

    return filt_data


def plot_fft(signal, sampling_rate, title="FFT Spectrum"):
    """
    Plot the FFT spectrum of a signal (1D or first channel of 2D).

    Parameters:
        signal (np.ndarray): Input signal (1D or 2D array).
        sampling_rate (float): Sampling frequency in Hz.
        title (str): Title of the plot.
    """
    # choose the first channel if there are more than one channel
    if signal.ndim > 1:
        signal = signal[0]

    # calculate FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    fft_vals = np.abs(np.fft.rfft(signal))

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_vals, label="Magnitude Spectrum")
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def group_files(file_list):
    """
    Groups and sorts .mat files based on their filename suffix.

    This function processes a list of file paths, grouping files that belong to the same trial
    by extracting the trial name prefix and the alphabetical suffix (e.g., 'a', 'b', 'c')
    before the '.mat' extension. It then sorts the files within each group according to their
    suffix letter to maintain the correct order for sequential processing.

    Parameters:
        file_list (list of str): List of full file paths to .mat files.

    Returns:
        dict: A dictionary where each key is a trial name prefix (str) and the value is a list
              of tuples (suffix_letter, file_path), sorted by suffix_letter. Files without suffix
              letters are grouped by their full filename as keys with empty suffix.
    """
    pattern = re.compile(r"^(.*?)([a-z])\.mat$")
    grouped = defaultdict(list)
    for f in file_list:
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if match:
            trial_name = match.group(1)
            suffix = match.group(2)
            grouped[trial_name].append((suffix, f))
        else:
            grouped[basename].append(("", f))

    for trial_name in grouped:
        grouped[trial_name].sort(key=lambda x: x[0])

    return grouped


def smooth_transition(
    arr1, arr2, smooth_points=50, window_size=5, plot_comparison=False
):
    """
    Smooth transitions between two 2D signals by applying moving average to
    the specified number of points around the junction.

    Parameters:
        arr1 (np.ndarray): First 2D array of shape (channels, time).
        arr2 (np.ndarray): Second 2D array of shape (channels, time).
        smooth_points (int): Number of points to smooth on each side of the junction.
        window_size (int): Window size for moving average (must be odd), number of points.
        plot_comparison (bool): Whether to plot before/after comparison.

    Returns:
        np.ndarray: Concatenated array with smooth transition.
    """

    if window_size % 2 != 1:
        raise ValueError("Window size must be odd for moving average.")

    # Calculate the transition region (smooth_points from each array)
    transition_start = arr1.shape[1] - smooth_points
    transition_end = arr1.shape[1] + smooth_points

    # Create the original concatenated array
    original = np.hstack([arr1, arr2])

    # Create a copy to apply smoothing
    smoothed = original.copy()

    # Apply moving average to each channel in the transition region
    for ch in range(arr1.shape[0]):
        # Get the transition region (including some context for smoothing)
        start_idx = max(0, transition_start - math.ceil(window_size / 2))
        end_idx = min(original.shape[1], transition_end + math.ceil(window_size / 2))

        # Smooth the extended region
        smoothed_region = smooth(original[ch, start_idx:end_idx], window_size)

        # Put back only the actual transition region we wanted to smooth
        put_start = transition_start
        put_end = transition_end
        smoothed[ch, put_start:put_end] = smoothed_region[
            math.ceil(window_size / 2) : math.ceil(window_size / 2)
            + (put_end - put_start)
        ]

    # Plot comparison if requested
    if plot_comparison:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot original
        ax1.plot(original.T, alpha=0.7)
        ax1.set_title("Before Smooth Transition")
        ax1.axvline(x=arr1.shape[1], color="r", linestyle="--", alpha=0.5)
        ax1.axvspan(transition_start, transition_end, color="r", alpha=0.1)
        ax1.set_ylabel("Amplitude")

        # Plot smoothed
        ax2.plot(smoothed.T, alpha=0.7)
        ax2.set_title("After Smooth Transition (Moving Average)")
        ax2.axvline(x=arr1.shape[1], color="g", linestyle="--", alpha=0.5)
        ax2.axvspan(transition_start, transition_end, color="g", alpha=0.1)
        ax2.set_xlabel("Time Samples")
        ax2.set_ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    return smoothed


def smooth(a, w, head=True, tail=True):
    """
    Matlab-like smooth function for 1D array

    Parameters:
        a: 1D array to be smoothed
        w: smoothing window size, must be odd number.
        head: whether to smooth the beginning of the array
        tail: whether to smooth the end of the array

    goes to https://stackoverflow.com/a/40443565/2366315
    """
    assert w % 2 == 1, "Need odd window size!"
    if len(a) < w:
        return a

    smoothed = np.convolve(a, np.ones(w, int), "valid") / w

    r = np.arange(1, w - 1, 2)
    if head:
        head = np.cumsum(a[: w - 1])[::2] / r
        smoothed = np.r_[head, smoothed]

    if tail:
        tail = (np.cumsum(a[:-w:-1])[::2] / r)[::-1]
        smoothed = np.r_[smoothed, tail]

    return smoothed


def get_sorted_mat_files(folder):
    """Return all .mat files in the given folder (sorted by filename in alphabetical order)."""
    # Find all .mat files in the folder
    files = glob.glob(os.path.join(folder, "*.mat"))
    # Sort files alphabetically by filename (not full path)
    files = sorted(files, key=lambda x: os.path.basename(x))
    return files


def find_artifact_zscore(
    x, threshold, fs, expand_ms=20, min_duration_ms=40, merge_gap_s=1.0
):
    """
    Detect artifact regions using modified Z-score method, with expansion,
    minimum duration filtering, and merging of close regions.

    Parameters:
    x : array-like
        Signal array.
    threshold : float
        Modified Z-score threshold for detecting outliers.
    fs : float
        Sampling rate of the signal (Hz).
    expand_ms : float
        Amount of time to expand around each detected point (ms).
    min_duration_ms : float
        Minimum duration for an artifact region to be kept (ms).
    merge_gap_s : float
        Maximum gap (in seconds) between two artifact regions to merge them.

    Returns:
    artifact_regions : list of tuples
        List of (start_idx, end_idx) for each artifact region.
    """
    # --- Step 1: Modified Z-score ---
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))

    if mad == 0:
        modified_z = np.zeros_like(x)
    else:
        modified_z = 0.6745 * (x - median_x) / mad

    outlier_idx = np.where(np.abs(modified_z) > threshold)[0]

    if len(outlier_idx) == 0:
        return []

    # --- Step 2: Expand each point ±expand_ms ---
    expand_samples = int(expand_ms / 1000 * fs)
    expanded_mask = np.zeros_like(x, dtype=bool)
    for idx in outlier_idx:
        start = max(0, idx - expand_samples)
        end = min(len(x), idx + expand_samples + 1)
        expanded_mask[start:end] = True

    # --- Step 3: Find contiguous artifact regions ---
    artifact_regions = []
    in_region = False
    for i, val in enumerate(expanded_mask):
        if val and not in_region:
            region_start = i
            in_region = True
        elif not val and in_region:
            region_end = i
            in_region = False
            artifact_regions.append((region_start, region_end))
    if in_region:
        artifact_regions.append((region_start, len(x)))

    # --- Step 4: Filter out short artifact regions ---
    min_samples = int(min_duration_ms / 1000 * fs)
    artifact_regions = [
        (start, end) for start, end in artifact_regions if (end - start) >= min_samples
    ]

    # --- Step 5: Merge close artifact regions ---
    merged_regions = []
    merge_gap_samples = int(merge_gap_s * fs)

    for region in artifact_regions:
        if not merged_regions:
            merged_regions.append(region)
        else:
            last_start, last_end = merged_regions[-1]
            curr_start, curr_end = region
            if curr_start - last_end <= merge_gap_samples:
                # Merge with previous region
                merged_regions[-1] = (last_start, max(last_end, curr_end))
            else:
                merged_regions.append(region)

    return merged_regions


def remove_artifacts_by_interpolation(
    x, regions, fs, pad_ms=10, fade_ms=5, interp_kind="cubic"
):
    """
    Remove artifact regions by replacing them with interpolation using neighboring data.

    Parameters
    ----------
    x : array
        Input signal
    regions : list of tuples
        Artifact regions [(start_idx, end_idx), ...]
    fs : float
        Sampling frequency (Hz)
    pad_ms : float
        Extend region by this many ms on both sides when choosing reference points
    fade_ms : float
        Apply fade-in/out to avoid discontinuities
    interp_kind : str
        'linear', 'quadratic', 'cubic' for interpolation type
    """
    x_clean = x.copy()
    n = len(x)
    pad = max(1, int(pad_ms / 1000 * fs))
    fade = max(1, int(fade_ms / 1000 * fs))

    for s, e in regions:
        s2 = max(0, s - pad)
        e2 = min(n, e + pad)

        # choose reference points outside artifact region
        left_idx = np.arange(max(0, s2 - pad), s2)
        right_idx = np.arange(e2, min(n, e2 + pad))

        if len(left_idx) < 2 or len(right_idx) < 2:
            # fallback: simple linear interpolation between edges
            left_val = x_clean[s2 - 1] if s2 - 1 >= 0 else 0.0
            right_val = x_clean[e2] if e2 < n else 0.0
            interp_vals = np.linspace(left_val, right_val, e2 - s2)
            x_clean[s2:e2] = interp_vals
        else:
            # build reference points
            ref_idx = np.concatenate([left_idx, right_idx])
            ref_vals = x_clean[ref_idx]

            # nonlinear interpolation
            try:
                if interp_kind == "cubic":
                    f = CubicSpline(ref_idx, ref_vals)
                else:
                    f = interp1d(
                        ref_idx, ref_vals, kind=interp_kind, fill_value="extrapolate"
                    )
                xi = np.arange(s2, e2)
                x_clean[s2:e2] = f(xi)
            except Exception:
                # fallback if spline fails
                xi = np.arange(s2, e2)
                x_clean[s2:e2] = np.interp(xi, ref_idx, ref_vals)

        # fade edges inside original region
        fade_in = np.linspace(0, 1, fade)
        fade_out = np.linspace(1, 0, fade)
        a = max(s, s2)
        b = min(e, e2)
        L = min(fade, b - a)
        if L > 0:
            x_clean[a : a + L] = x_clean[a : a + L] * fade_in[:L] + x[a : a + L] * (
                1 - fade_in[:L]
            )
            x_clean[b - L : b] = x_clean[b - L : b] * fade_out[-L:] + x[b - L : b] * (
                1 - fade_out[-L:]
            )

    return x_clean


def remove_artifacts_with_envelope(
    x, artifact_regions, fs, extend_s=0.5, smooth_s=0.1, method="abs_smooth"
):
    """
    Remove artifacts by extending artifact regions, computing envelope, subtracting it,
    and applying smoothing at the edges to avoid abrupt jumps.

    Parameters
    ----------
    x : 1D numpy array
        Input signal
    artifact_regions : list of tuples
        List of artifact intervals [(start_idx, end_idx), ...]
    fs : float
        Sampling rate (Hz)
    extend_s : float
        Time in seconds to extend artifact regions forward
    smooth_s : float
        Time in seconds for smoothing at edges
    method : str
        "hilbert" to use Hilbert envelope, "abs_smooth" to use absolute + moving average

    Returns
    -------
    x_clean : numpy array
        Signal after artifact removal
    """
    x_clean = x.copy()
    n = len(x)

    extend_samples = int(extend_s * fs)
    smooth_samples = int(smooth_s * fs)

    for start, end in artifact_regions:
        # --- Extend artifact region forward ---
        end_extended = min(n, end + extend_samples)

        # --- Extract the segment ---
        segment = x_clean[start:end_extended]

        # --- Compute envelope ---
        if method == "hilbert":
            segment_env = np.abs(hilbert(segment))
        elif method == "abs_smooth":
            win = max(1, int(0.01 * fs))  # 10 ms window
            segment_env = np.convolve(np.abs(segment), np.ones(win) / win, mode="same")
        else:
            raise ValueError("method must be 'hilbert' or 'abs_smooth'")

        # --- Apply smoothing at edges ---
        if smooth_samples > 0:
            fade_in = np.linspace(0, 1, smooth_samples)
            fade_out = np.linspace(1, 0, smooth_samples)

            segment_env[:smooth_samples] *= fade_in
            segment_env[-smooth_samples:] *= fade_out

        # --- Subtract envelope from the segment ---
        x_clean[start:end_extended] -= segment_env

    return x_clean


def artifact_removal_wavelet(x, wavelet="haar", level=6):
    """
    SWT-based artifact removal, preserving slow wave, fiber and ripple
    Args:
        x: input signal (1D numpy array)
        wavelet: wavelet name
        level: number of SWT decomposition levels
    Returns:
        data_new: cleaned signal
        coeffs_new: thresholded SWT coefficients
        id_D: list of arrays of indices of thresholded detail coefficients
    """

    # --- Step 1: Padding ---
    n = len(x)
    next_pow2 = 2 ** int(np.ceil(np.log2(n)))
    x_padded = np.pad(x, (0, next_pow2 - n), mode="constant")

    # --- Step 2: SWT decomposition ---
    coeffs = pywt.swt(x_padded, wavelet, level=level)

    # --- Step 3: Approximation coefficient thresholding ---
    cA_last, _ = coeffs[-1]

    k1 = 3
    sigma = np.median(np.abs(cA_last)) / 0.6745  # MAD（median absolute deviation ）
    T = k1 * np.sqrt(2 * np.log10(len(cA_last)) * sigma**2)  # soft threshold
    id_A = np.where(np.abs(cA_last) > T)[0]

    # Garrote thresholding
    cA_new = cA_last.copy()
    cA_new[id_A] = T**2 / cA_last[id_A]

    # --- Step 4: Detail coefficient thresholding ---
    D_new = []
    id_D = []
    for i, (cA, cD) in enumerate(coeffs, start=1):
        # Determine threshold multiplier k2
        if i in [2, 5, 6]:  # ripple layers,so high threshold
            k2 = 3
        else:
            k2 = 1

        sigma_sq = np.median(np.abs(cD)) / 0.6745
        Th = k2 * np.sqrt(2 * np.log10(len(cD)) * sigma_sq**2)  # ** 2
        idx = np.where(np.abs(cD) > Th)[0]

        # Garrote thresholding
        cD_new = cD.copy()
        cD_new[idx] = Th**2 / cD[idx]

        D_new.append(cD_new)
        id_D.append(idx)

    # --- Step 5: Reconstruct signal ---
    coeffs_new = []
    for i, (cA, cD) in enumerate(coeffs):
        if i == len(coeffs) - 1:
            coeffs_new.append((cA_new, D_new[i]))
        else:
            coeffs_new.append((cA, D_new[i]))

    data_new = pywt.iswt(coeffs_new, wavelet)
    data_new = data_new[:n]

    return data_new


'''
def artifact_removal_eemd_ica_segmented(x, fs, n_imfs=10, ica_components=None,
                                        artifact_indices_threshold=5, segment_sec=60):
    """
    Remove artifacts from long single-channel LFP using EEMD + ICA with segment processing.

    Parameters
    ----------
    x : array-like
        Input LFP signal (1D array)
    fs : float
        Sampling rate (Hz)
    n_imfs : int
        Number of IMFs to compute with EEMD
    ica_components : int or None
        Number of ICA components to extract. If None, set equal to number of IMFs
    artifact_indices_threshold : float
        Threshold in standard deviations to classify ICA components as artifact
    segment_sec : float
        Length of each segment in seconds for processing (default 60s)

    Returns
    -------
    clean_signal : np.ndarray
        LFP signal with artifacts removed
    artifact_regions : list of tuples
        List of (start_idx, end_idx) corresponding to removed artifact regions
    """
    x = np.asarray(x)
    n_samples = len(x)
    segment_samples = int(segment_sec * fs)

    clean_signal = np.zeros_like(x)
    artifact_mask = np.zeros_like(x, dtype=bool)

    for start in range(0, n_samples, segment_samples):
        end = min(start + segment_samples, n_samples)
        segment = x[start:end]

        # --- Step 1: EEMD decomposition ---
        eemd = EEMD()
        eemd.noise_seed(42)
        imfs = eemd.eemd(segment, max_imf=n_imfs)

        # --- Step 2: ICA on IMFs ---
        imfs_T = imfs.T
        n_components = ica_components if ica_components is not None else imfs.shape[0]
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        S_ = ica.fit_transform(imfs_T)
        A_ = ica.mixing_

        # --- Step 3: Detect artifact ICs ---
        artifact_ICs = []
        for i in range(S_.shape[1]):
            comp = S_[:, i]
            zscore = (comp - np.mean(comp)) / np.std(comp)
            if np.max(np.abs(zscore)) > artifact_indices_threshold:
                artifact_ICs.append(i)

        # --- Step 4: Zero out artifact ICs and reconstruct ---
        S_clean = S_.copy()
        S_clean[:, artifact_ICs] = 0
        imfs_clean = S_clean @ A_.T
        imfs_clean = imfs_clean.T
        clean_signal[start:end] = np.sum(imfs_clean, axis=0)

        # --- Step 5: Mark artifact points ---
        if len(artifact_ICs) > 0:
            artifact_signal = np.sum(imfs[artifact_ICs, :], axis=0)
            artifact_mask[start:end][np.abs(artifact_signal) > np.std(artifact_signal)] = True

    # --- Step 6: Merge artifact mask into regions ---
    artifact_regions = []
    in_region = False
    for i, val in enumerate(artifact_mask):
        if val and not in_region:
            region_start = i
            in_region = True
        elif not val and in_region:
            region_end = i
            in_region = False
            artifact_regions.append((region_start, region_end))
    if in_region:
        artifact_regions.append((region_start, len(x)))

    return clean_signal, artifact_regions
'''


'''
def artifact_removal_wavelet(x, wavelet='haar', level=6):
    """
    SWT-based artifact removal
    Args:
        x: input signal (1D numpy array)
        wavelet: wavelet name
        level: number of SWT decomposition levels
    Returns:
        data_new: cleaned signal
        coeffs_new: thresholded SWT coefficients
        id_A: indices of thresholded approximation coefficients
        id_D: list of arrays of indices of thresholded detail coefficients
    """
    # --- Step 1: Padding to next power of 2 (for SWT) ---
    n = len(x)
    next_pow2 = 2 ** int(np.ceil(np.log2(n)))
    x_padded = np.pad(x, (0, next_pow2 - n), mode='constant')

    # --- Step 2: SWT decomposition ---
    coeffs = pywt.swt(x_padded, wavelet, level=level)

    # --- Step 3: Approximation coefficient thresholding (lowest freq layer) ---
    cA_last, _ = coeffs[-1]
    A_old = cA_last.copy()
    A_new = cA_last.copy()

    sigma = np.median(np.abs(A_new)) / 0.6745 # MAD（median absolute deviation ）
    avg_ratio = np.max(np.abs(A_new))
    thr_ratio = 3
    if avg_ratio > 2 * thr_ratio:
        k1 = 0.5
    elif thr_ratio < avg_ratio <= 2 * thr_ratio:
        k1 = 0.75
    else:
        k1 = 1
    T = k1 * np.sqrt(2 * np.log10(len(A_new)) * sigma ** 2) # soft threshold
    id_A = np.where(np.abs(A_new) > T)[0]
    A_new[id_A] = 0


    # --- Step 4: Detail coefficient thresholding ---
    D_new = []
    id_D = []
    for i, (cA, cD) in enumerate(coeffs, start=1):
        # Determine threshold multiplier k2
        if i in [2, 3]:  # ripple layers,so high threshold
            k2 = 3
        else:
            k2 = 1

        sigma_sq = np.median(np.abs(cD)) / 0.6745
        Th = k2 * np.sqrt(2 * np.log10(len(cD)) * sigma_sq ** 2)
        idx = np.where(np.abs(cD) > Th)[0]

        # Garrote thresholding
        cD_new = cD.copy()
        cD_new[idx] = Th ** 2 / cD[idx]

        D_new.append(cD_new)
        id_D.append(idx)

    # --- Step 5: Reconstruct signal using ISWT ---
    coeffs_new = [(cA, cD_new) for (_, _), cD_new in zip(coeffs, D_new)]
    data_new = pywt.iswt(coeffs_new, wavelet)
    data_new = data_new[:n]  # remove padding
    data_new = data_new - np.mean(data_new)

    return data_new
'''


'''

def plot_signal(original, filtered, fs, window_sec = 5.0):
    """
        Interactive plot with time window scrolling for original and filtered signals.

        Parameters:
            original (np.ndarray): Original signal (1D).
            filtered (np.ndarray): Filtered signal (same shape as original).
            fs (float): Sampling rate in Hz.
            window_sec (float): Duration (in seconds) of the visible time window.
        """

    total_len = len(original)
    total_time = total_len / fs
    win_len = int(window_sec * fs)
    t = np.arange(total_len) / fs

    start_idx = 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6))
    plt.subplots_adjust(bottom=0.25, hspace=0.4)

    line1, = ax1.plot(t[:win_len], original[:win_len], lw=1.2, color='royalblue')
    ax1.set_title("Original Signal", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Amplitude (μV)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    line2, = ax2.plot(t[:win_len], filtered[:win_len], lw=1.2, color='seagreen')
    ax2.set_title("Filtered Signal", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Amplitude (μV)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)


    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 0, total_time - window_sec, valinit=0, valstep=window_sec / 10)

    def update(val):
        start = int(slider.val * fs)
        end = min(start + win_len, total_len)
        line1.set_xdata(t[start:end])
        line1.set_ydata(original[start:end])
        ax1.set_xlim(t[start], t[end - 1])

        line2.set_xdata(t[start:end])
        line2.set_ydata(filtered[start:end])
        ax2.set_xlim(t[start], t[end - 1])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax_left = plt.axes([0.05, 0.1, 0.05, 0.04])
    btn_left = Button(ax_left, '←')
    btn_left.on_clicked(lambda event: slider.set_val(max(slider.val - window_sec / 10, 0)))

    ax_right = plt.axes([0.88, 0.1, 0.05, 0.04])
    btn_right = Button(ax_right, '→')
    btn_right.on_clicked(lambda event: slider.set_val(min(slider.val + window_sec / 10, total_time - window_sec)))

    plt.show()
'''
