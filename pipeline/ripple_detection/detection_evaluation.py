# Some functions and definitions used to evaluate the performance of detection method: event-based strategy and
# time_interval_based strategy

import numpy as np
import pandas as pd


def get_binary_predictions(y_prob, threshold, min_distance, min_width):
    """
    Convert a probability trace into a binary event trace with post-processing.

    Parameters
    ----------
    y_prob : ndarray, shape (T,)
        Model output probability at each time point. Values should be in [0, 1]

    threshold : float
        Probability threshold for event detection
        Time points with probability >= threshold are treated as candidate event points.

    min_distance : int
        Minimum allowed gap (in time points) between two events
        If the gap (start_next - end_prev) between two events is smaller than this value, they will be merged into one event

    min_width : int
        Minimum event duration (in time points)
    Events shorter than this value will be removed.

    Returns
    -------
    y_binary : ndarray, shape (T,)
        Binary prediction trace after post-processing (0/1).
     pred_peaks : ndarray, shape (N_events,)
        Peak indices (in original time index) for each detected event.
    """
    y_prob = np.asarray(y_prob).flatten()
    T = y_prob.shape[0]

    # thresholding
    binary = (y_prob >= threshold).astype(np.int32)

    # find contiguous segments
    # diff == 1 -> event starts, diff == -1 -> event ends
    padded = np.concatenate([[0], binary, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # end is exclusive

    raw_events = []
    for start, end in zip(starts, ends):
        raw_events.append([start, end])

    # merge events that are too close
    merged_events = []
    for start, end in raw_events:
        if not merged_events:
            merged_events.append([start, end])
        else:
            prev_s, prev_e = merged_events[-1]
            gap = start - prev_e
            if gap < min_distance:
                # Merge by extending the previous event's end
                merged_events[-1][1] = end
            else:
                merged_events.append([start, end])

    # filter out events that are too short
    final_events = []
    for start, end in merged_events:
        if (end - start) >= min_width:
            final_events.append([start, end])

    # build final binary trace
    y_binary = np.zeros(T, dtype=np.int32)
    detected_peaks = []
    for start, end in final_events:
        y_binary[start:end] = 1

        seg = y_prob[start:end]  # end is exclusive
        if seg.size == 0:
            continue

        max_val = float(seg.max())
        max_pos = np.where(seg == max_val)[0]  # indices within seg
        pos = int(max_pos[len(max_pos) // 2])

        detected_peaks.append(start + pos)

    return y_binary, np.asarray(detected_peaks)


def get_tps_fps_fns_event_based(
    y_prob, ripple_onset, ripple_offset, detected_peaks, decimals=3
):
    """
    Event based evaluations using one to one peak-to-event matching

    Parameters
    ----------
    y_prob : np.ndarray,shape [T]
        Predicted probability trace
    ripple_onset : np.ndarray,shape [N]
        Ground-truth ripple onset indices
    ripple_offset : np.ndarray,shape [N]
        Ground-truth ripple offset indices
    detected_peaks: np.ndarray,shape [N]
        peaks of the detected ripples
        for machine learning method, it is the peak of probability traces
        for threshold method, it is the
    decimals : int
        Rounding for metrics.

    Returns
    -------
    tps : np.ndarray
    fps : np.ndarray
    fns : np.ndarray
    stats : pd.DataFrame
        TP/FP/FN and precision/recall/F1.
    """

    y_prob = np.asarray(y_prob).reshape(-1)
    ripple_onset = np.asarray(ripple_onset).astype(int).reshape(-1)
    ripple_offset = np.asarray(ripple_offset).astype(int).reshape(-1)
    if ripple_onset.size != ripple_offset.size:
        raise ValueError("ripple_onset and ripple_offset must have the same length.")

    # handle no GT case or no prediction case
    if ripple_onset.size == 0 or detected_peaks.size == 0:
        tps = np.array([], dtype=int)
        fps = detected_peaks.copy()
        fns = (
            np.arange(ripple_onset.size)
            if ripple_onset.size > 0
            else np.array([], dtype=int)
        )

        TP, FP, FN = 0, len(fps), len(fns)
        precision = 0.0
        recall = 0.0
        f1 = 0.0

        stats = pd.DataFrame(
            [
                [
                    TP,
                    FP,
                    FN,
                    FP + FN,
                    np.round(precision, decimals),
                    np.round(recall, decimals),
                    np.round(f1, decimals),
                ]
            ],
            columns=["TP", "FP", "FN", "FP+FN", "precision", "recall", "F_1"],
        )
        return tps, fps, fns, stats

    # one-to-one matching to find TPs
    tps = []
    fns = []

    # Iterate through each GT event to find the best matching peak
    for k, (s, e) in enumerate(zip(ripple_onset, ripple_offset)):
        inside_mask = (detected_peaks >= s) & (detected_peaks <= e)
        inside_peak_indices = np.where(inside_mask)[0]

        if inside_peak_indices.size == 0:
            fns.append(k)  # No peak found inside -> FN
            continue

        # Of the peaks inside, find the one with the highest probability
        best_peak_local_idx = inside_peak_indices[
            np.argmax(y_prob[detected_peaks[inside_peak_indices]])
        ]

        # Add the best peak's *value* (timepoint) to the TP list
        tps.append(detected_peaks[best_peak_local_idx])

    # Ensure TPs are unique, in case one peak is the best for two overlapping GTs
    tps = np.asarray(sorted(set(tps)), dtype=int)

    # Any detected peak that is NOT a True Positive is a False Positive.
    # This is much cleaner and more efficient.
    fps = np.setdiff1d(detected_peaks, tps, assume_unique=True)

    # Recalculate FNs based on unique TPs
    # A GT event is a FN if no TP peak falls within its boundaries.
    fns = []
    for k, (s, e) in enumerate(zip(ripple_onset, ripple_offset)):
        if not np.any((tps >= s) & (tps <= e)):
            fns.append(k)

    fns = np.asarray(fns, dtype=int)

    # metrics
    TP = tps.size
    FP = fps.size
    FN = fns.size  # FN is the number of GT events, not timepoints

    precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = (
        0.0
        if (precision + recall) == 0
        else (2 * precision * recall) / (precision + recall)
    )

    stats = pd.DataFrame(
        [
            [
                TP,
                FP,
                FN,
                FP + FN,
                np.round(precision, decimals),
                np.round(recall, decimals),
                np.round(f1, decimals),
            ]
        ],
        columns=["TP", "FP", "FN", "FP+FN", "precision", "recall", "F_1"],
    )

    return tps, fps, fns, stats


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int):
    """
    IoU of two inclusive 1D intervals [a_start, a_end] and [b_start, b_end].
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start + 1)

    len_a = a_end - a_start + 1
    len_b = b_end - b_start + 1
    union = len_a + len_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def get_tps_fps_fns_time_interval_based(
    y_prob,
    y_binary,
    ripple_onset,
    ripple_offset,
    gt_peaks,
    iou_threshold=0.3,
    peak_tol=10,
    decimals=3,
):
    """
    Time-interval–based evaluation using IoU matching .

    Parameters
    ----------
    y_binary : np.ndarray, shape [T]
        Binarized predicted trace (0/1)
    ripple_onset : np.ndarray, shape [N]
        Ground-truth ripple onset indices.
    ripple_offset : np.ndarray, shape [N]
        Ground-truth ripple offset indices (inclusive).
    iou_threshold : float
        IoU threshold for counting a match as TP.
    decimals : int
        Number of decimals for metric rounding.

    Returns
    -------
    tps : np.ndarray, shape [K, 2]
        Matched predicted intervals as [start, end] (inclusive).
    fps : np.ndarray, shape [P, 2]
        Unmatched predicted intervals as [start, end] (inclusive).
    fns : np.ndarray, shape [Q]
        Indices of unmatched GT events (indices into ripple_onset/ripple_offset).
    stats : pd.DataFrame
        DataFrame containing TP/FP/FN and precision/recall/F1.
    """

    y_binary = np.asarray(y_binary).reshape(-1).astype(np.uint8)
    T = y_binary.size

    ripple_onset = np.asarray(ripple_onset).astype(int).reshape(-1)
    ripple_offset = np.asarray(ripple_offset).astype(int).reshape(-1)
    if ripple_onset.size != ripple_offset.size:
        raise ValueError("ripple_onset and ripple_offset must have the same length.")

    # build intervals
    padded = np.concatenate([[0], y_binary, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = (
        np.where(diff == -1)[0] - 1
    )  # inclusive, because offset in the ground truth is also inclusive
    pred_intervals = (
        np.stack([starts, ends], axis=1)
        if starts.size > 0
        else np.zeros((0, 2), dtype=int)
    )
    gt_intervals = (
        np.stack([ripple_onset, ripple_offset], axis=1)
        if ripple_onset.size > 0
        else np.zeros((0, 2), dtype=int)
    )

    M = pred_intervals.shape[0]
    N = gt_intervals.shape[0]

    # predicted peaks (argmax of y_prob within each predicted interval)
    pred_peaks = np.zeros(M, dtype=int)
    for i in range(M):
        start, end = pred_intervals[i]
        if end >= start:
            pred_peaks[i] = start + np.argmax(y_prob[start : end + 1])
        else:
            pred_peaks[i] = start

    # ---- default assignments (cover all edge cases) ----
    matched_pred = np.zeros(M, dtype=bool)
    matched_gt = np.zeros(N, dtype=bool)

    # ---- match only if both sides exist ----
    if M > 0 and N > 0:
        # IoU matrix
        iou_mat = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            ps, pe = map(int, pred_intervals[i])
            for j in range(N):
                gs, ge = map(int, gt_intervals[j])
                iou_mat[i, j] = interval_iou(ps, pe, gs, ge)

        # candidate pairs: IoU + peak difference
        cand = []
        for i in range(M):
            for j in range(N):
                if iou_mat[i, j] < iou_threshold:
                    continue
                if abs(pred_peaks[i] - gt_peaks[j]) > peak_tol:
                    continue
                cand.append((i, j))

        cand = np.asarray(cand, dtype=int)

        if cand.size > 0:
            cand_ious = iou_mat[cand[:, 0], cand[:, 1]]
            cand = cand[np.argsort(-cand_ious)]  # descending IoU

            for pred_idx, gt_idx in cand:
                if matched_pred[pred_idx] or matched_gt[gt_idx]:
                    continue
                matched_pred[pred_idx] = True
                matched_gt[gt_idx] = True

    # ---- derive TP/FP/FN from matches ----
    tps = pred_intervals[matched_pred]
    fps = pred_intervals[~matched_pred]
    fns = np.where(~matched_gt)[0].astype(int)

    # ---- metrics ----
    TP, FP, FN = int(tps.shape[0]), int(fps.shape[0]), int(fns.size)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )

    stats = pd.DataFrame(
        [
            [
                TP,
                FP,
                FN,
                FP + FN,
                np.round(precision, decimals),
                np.round(recall, decimals),
                np.round(f1, decimals),
            ]
        ],
        columns=["TP", "FP", "FN", "FP+FN", "precision", "recall", "F_1"],
    )

    return tps, fps, fns, stats


'''
def get_tps_fps_fns_time_interval_based(y_binary,ripple_onset,ripple_offset,iou_threshold = 0.3,decimals = 3):
    """
    Time-interval–based evaluation using IoU matching .

    Parameters
    ----------
    y_binary : np.ndarray, shape [T]
        Binarized predicted trace (0/1)
    ripple_onset : np.ndarray, shape [N]
        Ground-truth ripple onset indices.
    ripple_offset : np.ndarray, shape [N]
        Ground-truth ripple offset indices (inclusive).
    iou_threshold : float
        IoU threshold for counting a match as TP.
    decimals : int
        Number of decimals for metric rounding.

    Returns
    -------
    tps : np.ndarray, shape [K, 2]
        Matched predicted intervals as [start, end] (inclusive).
    fps : np.ndarray, shape [P, 2]
        Unmatched predicted intervals as [start, end] (inclusive).
    fns : np.ndarray, shape [Q]
        Indices of unmatched GT events (indices into ripple_onset/ripple_offset).
    stats : pd.DataFrame
        DataFrame containing TP/FP/FN and precision/recall/F1.
    """

    y_binary = np.asarray(y_binary).reshape(-1).astype(np.uint8)
    T = y_binary.size

    ripple_onset = np.asarray(ripple_onset).astype(int).reshape(-1)
    ripple_offset = np.asarray(ripple_offset).astype(int).reshape(-1)
    if ripple_onset.size != ripple_offset.size:
        raise ValueError("ripple_onset and ripple_offset must have the same length.")

    # build intervals
    padded = np.concatenate([[0], y_binary, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1 # inclusive, because offset in the ground truth is also inclusive
    pred_intervals = np.stack([starts, ends], axis=1) if starts.size > 0 else np.zeros((0, 2), dtype=int)
    gt_intervals = np.stack([ripple_onset, ripple_offset], axis=1) if ripple_onset.size > 0 else np.zeros((0, 2), dtype=int)

    M = pred_intervals.shape[0]
    N = gt_intervals.shape[0]

    # ---- default assignments (cover all edge cases) ----
    matched_pred = np.zeros(M, dtype=bool)
    matched_gt = np.zeros(N, dtype=bool)

    # ---- match only if both sides exist ----
    if M > 0 and N > 0:
        # IoU matrix
        iou_mat = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            ps, pe = map(int, pred_intervals[i])
            for j in range(N):
                gs, ge = map(int, gt_intervals[j])
                iou_mat[i, j] = interval_iou(ps, pe, gs, ge)

        # candidate pairs
        cand = np.argwhere(iou_mat >= iou_threshold)  # [pred_idx, gt_idx]
        if cand.size > 0:
            cand_ious = iou_mat[cand[:, 0], cand[:, 1]]
            cand = cand[np.argsort(-cand_ious)]  # descending IoU

            for pred_idx, gt_idx in cand:
                if matched_pred[pred_idx] or matched_gt[gt_idx]:
                    continue
                matched_pred[pred_idx] = True
                matched_gt[gt_idx] = True

    # ---- derive TP/FP/FN from matches ----
    tps = pred_intervals[matched_pred]
    fps = pred_intervals[~matched_pred]
    fns = np.where(~matched_gt)[0].astype(int)

    # ---- metrics ----
    TP, FP, FN = int(tps.shape[0]), int(fps.shape[0]), int(fns.size)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    stats = pd.DataFrame(
        [[TP, FP, FN, FP + FN,
          np.round(precision, decimals),
          np.round(recall, decimals),
          np.round(f1, decimals)]],
        columns=["TP", "FP", "FN", "FP+FN", "precision", "recall", "F_1"],
    )

    return tps, fps, fns, stats
'''
