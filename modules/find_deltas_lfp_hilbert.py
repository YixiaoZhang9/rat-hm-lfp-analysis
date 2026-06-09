import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from modules.lfp_artifact_MAD_detection import mad_artifact_detector


def intervals_to_mask(intervals, timestamps):
    mask = np.zeros_like(timestamps, dtype=bool)
    for start, end in intervals:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def robust_zscore(x):
    """
    Robust z-score using median absolute deviation (MAD).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # avoid divide-by-zero
    if mad < 1e-10:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def filter_lfp(lfp, fs, freq_range):
    # Bandpass filter
    b, a = butter(4, np.array(freq_range) / (fs / 2), btype="band")
    return filtfilt(b, a, lfp)


def find_deltas_lfp_hilbert(
    raw_signal,
    signal,
    fs=1000,
    thresholds=(1, 2, 0, 1.5),
    durations=(150, 500),
    f_plot=False,
    plot_range=None,
):
    """
    Find cortical delta waves (0.5–4 Hz).

    Parameters
    ----------
    raw_signal: 1D array, LFP / EEG signal
    signal : 1D array, filtered LFP / EEG signal
    fs : float, Sampling rate (Hz)
    thresholds : tuple, (lowPeak, highPeak, lowTrough, highTrough)
    durations : tuple, (minDuration, maxDuration) in ms

    Returns
    -------
    delta : ndarray (N x 6)[start_index, peak_index, end_index, start_z, peak_z, end_z]
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

    # set thresholds
    lowPeak, highPeak, lowTrough, highTrough = thresholds
    minDuration, maxDuration = durations

    # convert ms to samples
    minSamples = int(minDuration * fs / 1000)
    maxSamples = int(maxDuration * fs / 1000)

    n = len(signal)
    idx = np.arange(n)

    z_raw = robust_zscore(signal)

    # hilbert transform
    analytic = hilbert(signal)
    phase = np.angle(analytic)

    # Detect troughs from phase wraps

    # +pi -> -pi wrap
    troughs = np.where(np.diff(phase) < -np.pi)[0] + 1
    if len(troughs) < 2:
        return np.empty((0, 6))

    # Build trough -> peak -> trough cycles
    triplets = []
    for i in range(len(troughs) - 1):
        a = troughs[i]
        c = troughs[i + 1]

        # duration constraint
        dur = c - a

        if dur < minSamples or dur > maxSamples:
            continue

        # artifact rejection
        if not np.all(valid_mask[a : c + 1]):
            continue

        segment = z_raw[a : c + 1]

        # peaks within cycle
        peaks, _ = find_peaks(segment)

        if len(peaks) == 0:
            continue

        # strongest peak
        peak_rel = peaks[np.argmax(segment[peaks])]

        b = a + peak_rel

        triplets.append([a, b, c])

    if len(triplets) == 0:
        return np.empty((0, 6))

    triplets = np.array(triplets)

    # Build delta matrix
    delta = np.zeros((len(triplets), 6))

    for i, (a, b, c) in enumerate(triplets):
        delta[i, 0] = idx[a]  # start index
        delta[i, 1] = idx[b]  # middle index
        delta[i, 2] = idx[c]  # end index

        delta[i, 3] = z_raw[a]
        delta[i, 4] = z_raw[b]
        delta[i, 5] = z_raw[c]

    # Amplitude conditions
    trough1 = delta[:, 3]
    peak = delta[:, 4]
    trough2 = delta[:, 5]

    trough = np.minimum(trough1, trough2)

    case1 = (peak > highPeak) & (trough <= -lowTrough)
    case2 = (peak >= lowPeak) & (trough < -highTrough)

    delta = delta[case1 | case2]

    if f_plot:
        if plot_range is None:
            plot_start = 0
            plot_end = len(signal)
        else:
            plot_start, plot_end = plot_range

        fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

        tt = np.arange(len(signal)) / fs

        # Raw signal
        ax = axes[0]

        raw_plot = filter_lfp(raw_signal, fs, [0.3, 30])

        ax.plot(
            tt[plot_start:plot_end],
            raw_plot[plot_start:plot_end],
            color="black",
            linewidth=1,
        )

        for start_s, end_s in artifact_intervals_s:
            ax.axvspan(start_s, end_s, color="gray", alpha=0.3)

        for row in delta:
            a = int(row[0])
            c = int(row[2])

            if c < plot_start or a > plot_end:
                continue

            ax.axvspan(a / fs, c / fs, color="red", alpha=0.2)

        ax.set_ylabel("Raw")
        ax.set_title("Raw signal + detected slow oscillations")

        # Delta filtered
        ax2 = axes[1]

        ax2.plot(tt[plot_start:plot_end], z_raw[plot_start:plot_end], color="black")

        ax2.scatter(troughs / fs, z_raw[troughs], color="blue", s=20, label="troughs")

        for row in delta:
            a = int(row[0])
            b = int(row[1])
            c = int(row[2])

            if c < plot_start or a > plot_end:
                continue

            ax2.axvspan(a / fs, c / fs, color="red", alpha=0.2)

            ax2.plot(
                [a / fs, b / fs, c / fs],
                [z_raw[a], z_raw[b], z_raw[c]],
                color="red",
                linewidth=2,
            )

        ax2.axhline(lowPeak, color="green", linestyle="--")

        ax2.axhline(highPeak, color="green")

        ax2.axhline(-lowTrough, color="purple", linestyle="--")

        ax2.axhline(-highTrough, color="purple")

        ax2.set_ylabel("Z-scored")
        ax2.legend()

        # Phase
        ax3 = axes[2]

        ax3.plot(tt[plot_start:plot_end], phase[plot_start:plot_end], color="black")

        ax3.scatter(troughs / fs, phase[troughs], color="red", s=20)

        ax3.set_ylabel("Phase")
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Hilbert phase")

        plt.tight_layout()

        plt.show()

    return delta
