import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import trapezoid

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


def find_deltas_lfp(
    raw_signal,
    signal,
    fs=1000,
    thresholds=(1, 2, 0, 1.5),
    durations=(150, 500),
    f_plot=False,
    plot_range=None,
):
    """
    Detect LFP delta events and extract waveform features.

    Parameters
    ----------
    raw_signal : array_like, shape (n_samples,)
        Unfiltered LFP signal used for artifact detection.

    signal : array_like, shape (n_samples,)
        Filtered LFP signal used for delta event detection and feature
        extraction. This is typically filtered in the delta range, e.g.
        0.5–4 Hz

    fs : float, default=1000
        Sampling frequency in Hz.

    thresholds : tuple of float, default=(1, 2, 0, 1.5)
        Amplitude thresholds in robust z-score units:
        (lowPeak, highPeak, lowTrough, highTrough).

        A candidate event is accepted if either:
            peak > highPeak and trough <= -lowTrough
        or:
            peak >= lowPeak and trough < -highTrough

    durations : tuple of float, default=(150, 500)
        Minimum and maximum event duration in milliseconds.

    f_plot : bool, default=False
        If True, plot the raw signal, z-scored filtered signal, artifacts,
        candidate extrema, and detected delta events.

    plot_range : tuple of int or None, default=None
        Optional sample range for plotting, given as (start_sample, end_sample).

    Returns
    -------
    delta_features : ndarray, shape (n_events, 10)
        Detected delta events and waveform features.

        Columns are:
        0. start_index
        1. peak_index
        2. end_index
        3. duration_s
        4. negative_amplitude
        5. positive_amplitude
        6. peak_to_peak_amplitude
        7. rising_slope
        8. decreasing_slope
        9. area under curve

    Copyright (C) 2012-2017 Michaël Zugaro, 2012-2015 Nicolas Maingret,

    the program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
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

    z_signal = robust_zscore(signal)

    # find the peaks and troughs
    # derivative-like signal
    signal_diff = np.diff(signal, prepend=signal[0])

    z_signal_diff = robust_zscore(signal_diff)

    # zero crossings
    up_crossing = (
        np.where((z_signal_diff[:-1] < 0) & (z_signal_diff[1:] >= 0))[0] + 1
    )  # up crossing
    down_crossing = (
        np.where((z_signal_diff[:-1] > 0) & (z_signal_diff[1:] <= 0))[0] + 1
    )  # down crossing

    if (
        len(down_crossing) > 0
        and len(up_crossing) > 0
        and down_crossing[0] < up_crossing[0]
    ):
        down_crossing = down_crossing[1:]

    n_events = min(len(up_crossing), len(down_crossing))
    if n_events < 2:
        return np.empty((0, 10))

    triplets = np.column_stack(
        [
            up_crossing[: n_events - 1],
            down_crossing[: n_events - 1],
            up_crossing[1:n_events],
        ]
    )

    # Strict artifact rejection
    # Reject candidate delta waves if they overlap with
    # an artifact interval or its surrounding buffer region
    # (artifact ± 1 s) to avoid filter ringing and
    # artifact-induced baseline distortions.

    artifact_padding_s = 1.0  # second

    expanded_artifact_mask = np.zeros_like(valid_mask)

    for start_s, end_s in artifact_intervals_s:
        start_s = max(0, start_s - artifact_padding_s)
        end_s = min(len(signal) / fs, end_s + artifact_padding_s)

        expanded_artifact_mask |= (t >= start_s) & (t <= end_s)

    keep_triplet = []

    for a, b, c in triplets:
        if np.any(expanded_artifact_mask[a : c + 1]):
            continue

        keep_triplet.append([a, b, c])

    if len(keep_triplet) == 0:
        return np.empty((0, 10))

    triplets = np.array(keep_triplet)

    delta = np.zeros((len(triplets), 6))

    for i, (a, b, c) in enumerate(triplets):
        delta[i, 0] = idx[a]  # start index
        delta[i, 1] = idx[b]  # middle index
        delta[i, 2] = idx[c]  # end index

        delta[i, 3] = z_signal[a]
        delta[i, 4] = z_signal[b]
        delta[i, 5] = z_signal[c]

    # Duration filter
    duration = delta[:, 2] - delta[:, 0]

    keep_duration = (duration >= minSamples) & (duration <= maxSamples)

    delta = delta[keep_duration]

    # Amplitude conditions
    trough1 = delta[:, 3]
    peak = delta[:, 4]
    trough2 = delta[:, 5]

    trough = np.minimum(trough1, trough2)

    case1 = (peak > highPeak) & (trough <= -lowTrough)
    case2 = (peak >= lowPeak) & (trough < -highTrough)

    delta = delta[case1 | case2]

    if len(delta) == 0:
        return np.empty((0, 10))

    # calculate the parameters
    delta_features = np.zeros((len(delta), 10))

    for i, row in enumerate(delta):
        a = int(row[0])
        b = int(row[1])
        c = int(row[2])

        event_signal = signal[a : c + 1]

        # Duration / s
        duration_s = (c - a) / fs

        # Amplitude features from filtered raw signal
        neg_amplitude = np.min(event_signal)
        pos_amplitude = np.max(event_signal)
        peak_to_peak_amplitude = pos_amplitude - neg_amplitude

        # Extrema indices within the event
        trough1_amplitude = event_signal[0]
        trough2_amplitude = event_signal[-1]

        # Rising slope: negative trough1 to positive peak
        rising_slope = (pos_amplitude - trough1_amplitude) / ((b - a) / fs)

        # Decreasing slope: positive peak to negative trough2
        decreasing_slope = abs((trough2_amplitude - pos_amplitude) / ((c - b) / fs))

        # area under curve
        auc = trapezoid(np.abs(event_signal), dx=1 / fs)

        delta_features[i, 0] = a
        delta_features[i, 1] = b
        delta_features[i, 2] = c
        delta_features[i, 3] = duration_s
        delta_features[i, 4] = neg_amplitude
        delta_features[i, 5] = pos_amplitude
        delta_features[i, 6] = peak_to_peak_amplitude
        delta_features[i, 7] = rising_slope
        delta_features[i, 8] = decreasing_slope
        delta_features[i, 9] = auc


    # plot
    if f_plot:
        if plot_range is None:
            plot_start = 0
            plot_end = len(signal)
        else:
            plot_start, plot_end = plot_range

        t = np.arange(len(signal)) / fs

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # raw signal
        ax = axes[0]
        filtered_raw_signal = filter_lfp(raw_signal, fs, [0.3, 30])

        ax.plot(
            t[plot_start:plot_end],
            filtered_raw_signal[plot_start:plot_end],
            color="black",
            linewidth=1,
        )

        # artifact regions
        for start_s, end_s in artifact_intervals_s:
            ax.axvspan(start_s, end_s, color="gray", alpha=0.3)

        # accepted delta events
        for row in delta:
            a = int(row[0])
            b = int(row[1])
            c = int(row[2])

            # only plot events in range
            if c < plot_start or a > plot_end:
                continue

            ax.axvspan(a / fs, c / fs, color="red", alpha=0.2)

            # ax.plot(
            #     [a / fs, b / fs, c / fs],
            #     [filtered_raw_signal[a], filtered_raw_signal[b], filtered_raw_signal[c]],
            #     color='red',
            #     linewidth=2
            # )

        ax.set_ylabel("Raw signal")
        ax.set_title("Raw signal(0.3-30Hz filtered) + detected delta waves")

        # z-scored signal

        ax2 = axes[1]
        ax2.plot(
            t[plot_start:plot_end],
            z_signal[plot_start:plot_end],
            color="black",
            linewidth=1,
        )

        # extrema
        ax2.scatter(
            up_crossing / fs,
            z_signal[up_crossing],
            color="blue",
            s=20,
            label="up crossing",
        )
        ax2.scatter(
            down_crossing / fs,
            z_signal[down_crossing],
            color="orange",
            s=20,
            label="down crossing",
        )

        # artifact regions
        for start_s, end_s in artifact_intervals_s:
            ax2.axvspan(start_s, end_s, color="gray", alpha=0.3)

        # detected delta
        for row in delta:
            a = int(row[0])
            b = int(row[1])
            c = int(row[2])

            if c < plot_start or a > plot_end:
                continue

            ax2.axvspan(a / fs, c / fs, color="red", alpha=0.2)

            ax2.plot(
                [a / fs, b / fs, c / fs],
                [z_signal[a], z_signal[b], z_signal[c]],
                color="red",
                linewidth=2,
            )

        # thresholds
        ax2.axhline(lowPeak, color="green", linestyle="--", alpha=0.5)

        ax2.axhline(highPeak, color="green", linestyle="-")

        ax2.axhline(-lowTrough, color="purple", linestyle="--", alpha=0.5)

        ax2.axhline(-highTrough, color="purple", linestyle="-")

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Z-scored signal")

        ax2.set_title(f"Detected Delta Waves | n={len(delta)}")

        ax2.legend()

        plt.tight_layout()
        plt.show()

    return delta_features
