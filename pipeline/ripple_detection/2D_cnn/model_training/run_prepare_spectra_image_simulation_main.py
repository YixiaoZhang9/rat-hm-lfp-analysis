import numpy as np
from pathlib import Path
from scipy.signal import hilbert,convolve
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from mne.time_frequency import tfr_array_stockwell
from PIL import Image
import matplotlib.cm as cm
from scipy.signal.windows import gaussian


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

def load_npz_trial(file):
    """
    Load one trial and return:
    - normalized LFP
    - soft label (consensus_trace)
    - upsampled scoring
    """
    d = np.load(file, allow_pickle=True)

    lfp = d["preprocessed_data"].astype(np.float32)
    y = d["consensus_trace"].astype(np.float32)
    scoring = d["scoring"].astype(np.int8)

    # upsample scoring: 1 Hz → 1000 Hz
    scoring = np.repeat(scoring, fs)

    # normalize signal
    lfp = zscore(lfp)

    return lfp, y, scoring


def extract_ripple_regions(y, threshold=0.5):
    """
    Convert soft label into ripple segments.

    Returns:
    - list of (start, end) indices
    """
    binary = (y >= threshold).astype(int)

    diff = np.diff(binary, prepend=0, append=0)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list(zip(starts, ends))


def compute_ripple_peaks(lfp, ripple_regions):
    """
    Compute ripple peak using Hilbert envelope
    """
    envelope = np.abs(hilbert(lfp))
    smoothed_envelope = smooth_signal(envelope, fs, 0.004)
    peaks = []
    for start, end in ripple_regions:
        if end > start:
            peak = start + np.argmax(smoothed_envelope[start:end])
        else:
            peak = start
        peaks.append(peak)

    return np.array(peaks)


def sample_negative_centers(signal_len, positive_centers,
                            window_len, n_neg, fs,
                            exclusion_ms=1000):
    """
    Sample negative centers away from ripple peaks.
    """
    half_len = window_len // 2
    exclusion = int(exclusion_ms / 1000 * fs)

    neg_centers = []
    tries = 0

    while len(neg_centers) < n_neg and tries < 5000:
        c = rng.integers(half_len, signal_len - half_len)

        if len(positive_centers) > 0:
            if np.any(np.abs(positive_centers - c) < exclusion):
                tries += 1
                continue

        neg_centers.append(c)
        tries += 1

    return np.array(neg_centers)


def stockwell_transform(segment, fs):
    """
    Convert signal segment into TF image
    """
    x = segment[None, None, :]

    power, _, _ = tfr_array_stockwell(
        x,
        sfreq=fs,
        fmin=f_min,
        fmax=f_max,
        width=1.0,
        n_jobs=1
    )

    tf = power[0].T.astype(np.float32)

    # smoothing
    tf = gaussian_filter(tf, sigma=1)

    # log scaling
    tf = np.log10(tf + 1e-12)

    # normalize to [0,1]
    tf_min, tf_max = tf.min(), tf.max()
    tf = (tf - tf_min) / (tf_max - tf_min + 1e-8)

    # convert to RGB
    img = cm.jet(tf)[..., :3]

    return img


#  main pipline
def generate_dataset(file_list, split="train"):

    pos_dir = out_dir / split / "yes"
    neg_dir = out_dir / split / "no"

    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_count = 0
    neg_count = 0

    window_len = int(window_ms / 1000 * fs)
    if window_len % 2 == 1:
        window_len += 1

    half_len = window_len // 2

    for file in file_list:
        print(f"Processing {file}")

        lfp, y, scoring = load_npz_trial(file)

        # only NREM segments
        nrem_mask = (scoring == 3)

        ripple_regions = extract_ripple_regions(y)

        ripple_regions = [
            (s, e) for s, e in ripple_regions
            if np.all(nrem_mask[s:e])
        ]


        ripple_peaks = compute_ripple_peaks(lfp, ripple_regions)
        n_pos_per_trial = 0
        # positive samples
        for i, center in enumerate(ripple_peaks):

            if center < half_len or center >= len(lfp) - half_len:
                continue

            jitter = int(rng.uniform(-jitter_ms, jitter_ms) / 1000 * fs)
            c = center + jitter

            start = c - half_len
            end = c + half_len

            segment = lfp[start:end]

            img = stockwell_transform(segment, fs)

            save_path = pos_dir / f"img_{pos_count:06d}.jpg"
            Image.fromarray((img * 255).astype(np.uint8)).save(save_path)

            pos_count += 1
            n_pos_per_trial += 1

        # negative samples
        neg_centers = sample_negative_centers(
            len(lfp),
            ripple_peaks,
            window_len,
            n_neg = n_pos_per_trial,
            fs = fs
        )

        for center in neg_centers:
            start = center - half_len
            end = center + half_len

            segment = lfp[start:end]

            img = stockwell_transform(segment, fs)

            save_path = neg_dir / f"img_{neg_count:06d}.jpg"
            Image.fromarray((img * 255).astype(np.uint8)).save(save_path)

            neg_count += 1

        print(f"Saved: {pos_count} pos, {neg_count} neg")



# configuration

fs = 1000
window_ms = 400          # window size for CNN
jitter_ms = 100          # jitter for positive samples

f_min = 100
f_max = 250

out_dir = Path("training_data/spectral_image")

rng = np.random.default_rng(42)
project_root = Path.cwd().resolve().parents[1]
data_root = project_root / "training_data"

train_list = sorted(data_root.glob("trial_00[1-4].npz"))
val_list   = sorted(data_root.glob("trial_005.npz"))

generate_dataset(train_list, split="train")
generate_dataset(val_list, split="val")

print("Done.")