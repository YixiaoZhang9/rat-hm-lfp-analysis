import emd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, get_window
from scipy.stats import skew


def short_time_energy(segment, fs, window_ms=15, window_type="hanning"):
    """
    Compute the short-time energy (STE) of a filtered signal using a customizable window.

    Parameters
    ----------
    signal : array_like
        1D filtered segment (e.g., ripple candidate).
    fs : float
        Sampling frequency in Hz.
    window_ms : float, optional
        Window length in milliseconds for computing short-time energy. Default is 15 ms.
    window_type : str, optional
        Window type: 'hanning', 'hamming', 'rectangular', 'blackman', etc. Default is 'hanning'.

    """
    segment = np.asarray(segment).flatten()

    # Compute window length in samples
    N = int(window_ms * fs / 1000)
    if N < 1:
        raise ValueError("Window length too short for the given sampling rate.")

    # Square the signal
    squared_segment = segment**2

    # Create normalized window
    if window_type.lower() == "rectangular":
        window = np.ones(N) / N
    else:
        window = get_window(window_type, N)
        window = window / np.sum(window)  # normalize

    # Convolve squared signal with window
    short_time_energy = np.convolve(squared_segment, window, mode="same")

    return short_time_energy


def spectral_centroid_short(segment, fs, window_type="hamming"):
    """
    Compute the spectral centroid of a short signal segment (e.g., single HFO event).

    Parameters
    ----------
    segment : array_like
        1D array representing the signal segment (HFO candidate).
    fs : float
        Sampling frequency in Hz.
    window_type : str, optional
        Type of window to apply before FFT (e.g., 'hamming', 'hanning', 'blackman'). Default is 'hamming'.

    Returns
    -------
    fc : float
        Spectral centroid in Hz (frequency corresponding to the center of the power spectrum).

    Notes
    -----
    - The function uses a window function to reduce spectral leakage.
    - The FFT length is set to the next power of 2 of the segment length, capped at 256 points to avoid excessive zero-padding.
    - If the segment is all zeros, the function returns 0.0.
    """
    segment = np.asarray(segment).flatten()
    L = len(segment)

    if L < 1:
        raise ValueError("Segment length must be at least 1 sample.")

    # Apply window
    window = get_window(window_type, L)
    xw = segment * window

    # Determine FFT length
    n_fft = min(2 ** int(np.ceil(np.log2(L))), 256)

    # Compute one-sided FFT
    X = np.fft.rfft(xw, n=n_fft)
    P = np.abs(X) ** 2  # power spectrum
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    # Avoid division by zero
    if np.sum(P) == 0:
        return 0.0

    # Compute spectral centroid
    fc = np.sum(freqs * P) / np.sum(P)
    return fc


def ripple_envelop_features(
    env_s, ripple_segment, ripple_segment_filt, fs, start_ind, end_ind, plot=False
):
    """
    Extract temporal and statistical features from an envelope segment.

    Parameters
    ----------
    env_s : ndarray
        Envelope of a ripple segment (already bandpass-filtered and Hilbert-transformed).
    fs : float
        Sampling rate in Hz.
    plot : bool, optional
        If True, plots the envelope with rise/fall times and peak marked.

    Returns
    -------
    features : dict
        Dictionary containing various envelope features:
        - duration_ms: Duration of the ripple segment in milliseconds
        - peak_env: Maximum value of the envelope
        - rms_env: Root mean square of the envelope
        - relative_rise_time: (20%-80% rise time)/total time
        - relative_fall_time: (80%-20% fall slope)/total time
        - env_skewness: Skewness of the envelope
        - peak_energy_fraction: Fraction of energy in the top 20% of envelope amplitudes
    """

    # ---------------Basic temporal features--------------

    baseline = np.min(env_s)
    env_rel = env_s - baseline  # relative amplitude

    peak_idx = np.argmax(env_s)
    peak_env = env_rel[peak_idx]
    duration_ms = len(env_s) / fs * 1000
    rms_env = np.sqrt(np.mean(env_rel**2))

    # ---------------Rise and fall times (20%-80% of peak)--------------

    # Rise time (left of peak)
    low_thresh = 0.2 * peak_env
    high_thresh = 0.8 * peak_env

    # Rise time (left of peak)
    lt = np.where(env_rel[:peak_idx] < low_thresh)[0]
    ht = np.where(env_rel[:peak_idx] > high_thresh)[0]
    if len(lt) > 0 and len(ht) > 0:
        relative_rise_time = (ht[0] - lt[-1]) / len(env_s)
        rise_start_idx = lt[-1]
        rise_end_idx = ht[0]
    else:
        relative_rise_time = 0
        rise_start_idx = rise_end_idx = None

    # Fall time (right of peak)
    rt = np.where(env_rel[peak_idx:] < low_thresh)[0]
    ht2 = np.where(env_rel[peak_idx:] > high_thresh)[0]
    if len(rt) > 0 and len(ht2) > 0:
        relative_fall_time = (rt[0] - ht2[-1]) / len(env_s)
        fall_start_idx = ht2[-1] + peak_idx
        fall_end_idx = rt[0] + peak_idx
    else:
        relative_fall_time = 0
        fall_start_idx = fall_end_idx = None

    # ---------------Statistical features--------------
    env_skew = skew(env_s)

    # ---------------Energy concentration (peakness)--------------
    sorted_env = np.sort(env_rel)
    top20_energy = np.sum(sorted_env[int(0.8 * len(sorted_env)) :])
    peak_energy_fraction = top20_energy / (np.sum(sorted_env) + 1e-12)

    features = {
        "duration_ms": duration_ms,
        "peak_env": peak_env,
        "rms_env": rms_env,
        "relative_rise_time": relative_rise_time,
        "relative_fall_time": relative_fall_time,
        "env_skewness": env_skew,
        "peak_energy_fraction": peak_energy_fraction,
    }

    # ----- Plotting (optional) -----
    if plot:
        # ======== Time axes (in milliseconds) ========
        t_full = np.arange(len(ripple_segment)) / fs * 1000  # Full signal time axis
        t_env = np.arange(len(env_s)) / fs * 1000  # Envelope time axis (relative)
        t_env_shifted = (
            t_full[start_ind] + t_env
        )  # Align envelope with full signal time

        # ======== Create one figure with two subplots ========
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # -----------------------------------------------------
        # (1) Top subplot: Raw ripple segment with highlighted region
        # -----------------------------------------------------
        axs[0].plot(t_full, ripple_segment, color="black", lw=1)
        axs[0].axvspan(
            t_full[start_ind],
            t_full[end_ind - 1],
            color="yellow",
            alpha=0.3,
            label="Ripple region",
        )
        axs[0].set_title("Raw Ripple Segment (with detected ripple window)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend(loc="upper right")

        # -----------------------------------------------------
        # (2) Bottom subplot: Filtered signal + Envelope + Annotations
        # -----------------------------------------------------
        axs[1].plot(
            t_full,
            ripple_segment_filt,
            color="gray",
            lw=1,
            alpha=0.6,
            label="Filtered Ripple",
        )
        axs[1].plot(t_env_shifted, env_s, color="blue", lw=2, label="Envelope")

        # ---- Mark key envelope features ----
        axs[1].axvline(
            t_env_shifted[peak_idx],
            color="magenta",
            linestyle="--",
            lw=1.5,
            label="Peak",
        )

        # Rise (20%→80%)
        if rise_start_idx is not None and rise_end_idx is not None:
            axs[1].axvline(
                t_env_shifted[rise_start_idx], color="green", linestyle="--", lw=1
            )
            axs[1].axvline(
                t_env_shifted[rise_end_idx], color="lime", linestyle="--", lw=1
            )

        # Fall (80%→20%)
        if fall_start_idx is not None and fall_end_idx is not None:
            axs[1].axvline(
                t_env_shifted[fall_start_idx], color="orange", linestyle="--", lw=1
            )
            axs[1].axvline(
                t_env_shifted[fall_end_idx], color="red", linestyle="--", lw=1
            )

        # ---- Feature summary text box ----
        text = "\n".join(
            [
                f"Duration: {duration_ms:.1f} ms",
                f"Peak env: {peak_env:.3f}",
                f"RMS env: {rms_env:.3f}",
                f"Rise time (rel): {relative_rise_time:.3f}",
                f"Fall time (rel): {relative_fall_time:.3f}",
                f"Skewness: {env_skew:.3f}",
                f"Peak energy frac: {peak_energy_fraction:.3f}",
            ]
        )
        axs[1].text(
            0.02,
            0.95,
            text,
            transform=axs[1].transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
        )

        axs[1].set_title("Filtered Ripple + Envelope + Extracted Features")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(loc="upper right")

        # ======== Layout adjustment and show ========
        plt.tight_layout()
        plt.show()

    return features


def ripple_time_domain_features(
    ripple_segment, ripple_segment_filt, fs, start_ind, end_ind, plot=False
):
    """
    Compute:
      1) peak_to_trough amplitude
      2) number of peaks
      3) zero crossing rate
      4) good cycle ratio
    for a ripple segment.
    """

    # --- slice the filtered segment ---
    rip = ripple_segment_filt[start_ind:end_ind]

    # ----- 1) peak-to-trough -----
    peak_to_trough = np.max(rip) - np.min(rip)

    # ----- 2) number of peaks -----
    peaks, _ = find_peaks(rip)
    num_peaks = len(peaks)

    # ----- 3) zero crossing rate -----
    signal_mid = rip - np.mean(rip)  # remove DC offset
    zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(signal_mid)))) / (
        2 * len(signal_mid)
    )

    # ----- 4) rate of good cycles (emd package) -----
    # https://emd.readthedocs.io/en/stable/emd_tutorials/03_cycle_ananlysis/emd_tutorial_03_cycle_01_detect.html#extract-imfs-find-cycles
    IP_full, IF_full, IA_full = emd.spectra.frequency_transform(
        ripple_segment_filt, fs, "hilbert"
    )
    IP_ripple = IP_full[start_ind:end_ind]

    cycle_vect_good = emd.cycles.get_cycle_vector(
        IP_ripple,
        return_good=True,
        phase_edge=np.pi / 4,
    )
    cycle_vect_all = emd.cycles.get_cycle_vector(
        IP_ripple,
        return_good=False,
        phase_edge=np.pi / 4,
    )

    # ratio of good cycle
    total = len(set(cycle_vect_all[cycle_vect_all >= 0]))
    good = len(set(cycle_vect_good[cycle_vect_good >= 0]))
    good_cycle_ratio = good / total if total > 0 else 0

    features = {
        "peak_to_trough": peak_to_trough,
        "num_peaks": num_peaks,
        "zero_crossing_rate": zero_crossing_rate,
        "good_cycle_ratio": good_cycle_ratio,
    }

    # ----- Plotting (optional) -----
    if plot:
        t_full = np.arange(len(ripple_segment)) / fs
        rip_full = np.full_like(t_full, np.nan, dtype=float)
        rip_IP_full = np.full_like(t_full, np.nan, dtype=float)
        rip_full[start_ind:end_ind] = rip
        rip_IP_full[start_ind:end_ind] = IP_ripple.squeeze()

        fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

        # --- Raw signal ---
        axs[0].plot(t_full, ripple_segment, color="black")
        axs[0].axvspan(
            t_full[start_ind], t_full[end_ind - 1], color="yellow", alpha=0.2
        )
        axs[0].set_title("Raw Ripple Segment")

        # --- Filtered ripple with cycles ---
        axs[1].plot(t_full, rip_full, color="blue")

        axs[2].plot(t_full, rip_IP_full, color="blue")

        # --- good cycles  ---
        unique_good_ids = set(cycle_vect_good[cycle_vect_good >= 0])
        for cid in unique_good_ids:
            idx = np.where(cycle_vect_good == cid)[0]
            start_t = (start_ind + idx[0]) / fs
            end_t = (start_ind + idx[-1]) / fs
            axs[1].axvspan(
                start_t,
                end_t,
                color="green",
                alpha=0.4,
                label="Good Cycle" if cid == list(unique_good_ids)[0] else "",
            )

        axs[1].set_title("Filtered Ripple with All Cycles and Good Cycles")
        axs[1].axvspan(
            t_full[start_ind], t_full[end_ind - 1], color="yellow", alpha=0.2
        )
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return features
