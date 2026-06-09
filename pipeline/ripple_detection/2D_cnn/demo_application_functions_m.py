import sys                                      # + path to fastai in root repo directory.
sys.path.insert(1, '../')
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
import scipy
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter,zoom
from scipy.stats import zscore
from mne.time_frequency import tfr_array_stockwell\
#https://mne.tools/stable/generated/mne.time_frequency.tfr_array_stockwell.html#mne.time_frequency.tfr_array_stockwell
from PIL import Image
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def load_npz_trial(file):
    d = np.load(file, allow_pickle=True)

    pre_lfp = d["preprocessed_data"].astype(np.float32)

    ripple_info = d["ripple_info"].item()
    meta = d["meta"].item()
    fs = int(meta["fs"])

    onset = np.asarray(ripple_info["onset"])
    offset = np.asarray(ripple_info["offset"])
    ripple_frequency = ripple_info["frequency"]


    #detect the peaks of true ripples
    ripple_onset = np.round(onset * fs).astype(int)
    ripple_offset = np.round(offset * fs).astype(int)
    envelope = np.abs(hilbert(pre_lfp))
    gt_peaks = []
    n = len(pre_lfp)
    for on, off in zip(ripple_onset, ripple_offset):
        on = max(0, on)
        off = min(n - 1, off)
        if off > on:
            peak_sample = on + np.argmax(envelope[on:off])
        else:
            peak_sample = on
        gt_peaks.append(peak_sample)

    ripple_peaks = np.round(gt_peaks).astype(int)

    true_ripple = np.column_stack((ripple_onset, ripple_offset))

    # normalization
    lfp = np.asarray(pre_lfp).reshape(-1)
    lfp_norm = zscore(lfp)
    return lfp_norm, true_ripple, ripple_peaks, fs,ripple_frequency

def stockwell_transform(segment, fs, fmin=100, fmax=250, width=1.0, decim=1, log_power=True, smooth_sigma = 1 ):
    """
    Compute Stockwell time-frequency image for one LFP segment.

    Input:
        segment : 1D array Input LFP segment, shape (n_times,)
        fs : float, Sampling frequency
        fmin, fmax : float, Frequency range of interest
        width : float, Stockwell width parameter
        decim : int, Temporal decimation factor
        log_power : bool, Whether to apply log10 transform
        smooth_sigma : float or None. Gaussian smoothing sigma. Example: 0.5 or 1.0

    Output:
        s_transform_44 : 2D float32 array, Resized time-frequency image
        freqs : 1D array, Frequency axis
        """
    x = np.asarray(segment, dtype=np.float32)[None, None, :]  # (1, 1, n_times)

    power,_,freqs = tfr_array_stockwell(x,fs,fmin=fmin,fmax=fmax,n_fft=None,width=width,decim=decim,n_jobs=1)

    # power shape: (n_epochs, n_freqs, n_times)
    # s_transform = np.abs(tfr_results[0]).squeeze())  # (n_freqs, n_times)
    # s_transform = s_transform.T # (n_times, n_freqs), the input in the papter is times bins* frequency bins
    # freqs = tfr_results[2]

    s_transform = power[0]  # (n_freqs, n_times)
    s_transform = s_transform.T.astype(np.float32) # (n_times, n_freqs), the input in the paper is times bins* frequency bins

    # optional smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        s_transform = gaussian_filter(s_transform, sigma=smooth_sigma)

    # apply log transition for CNN input
    if log_power:
        s_transform = np.log10(s_transform+ 1e-12)
    raw_s_transform = s_transform.copy()

    # Normalize to [0, 1] for colormap
    tf_min, tf_max = s_transform.min(), s_transform.max()
    if tf_max > tf_min:
        s_transform_norm = (s_transform - tf_min) / (tf_max - tf_min)
    else:
        s_transform_norm = np.zeros_like(s_transform, dtype=np.float32)

    # Apply jet colormap to mimic original paper style
    img_rgb = cm.jet(s_transform_norm)[..., :3].astype(np.float32)  # shape: (H, W, 3)

    return img_rgb, freqs

def make_spectra_image_files(data, time, out_dir, fs, segment_ms, overlap, f_min = 100, f_max = 250,
                             width = 1.0, decim = 1, log_power = True, smooth_sigma = 1):
    """
    Given 1D signal data and time vector, compute Stockwell TF image for each segment,
    save each image to disk, and return start/stop time dictionaries.

    Input:
        data : 1D array
            Signal values
        time : 1D array
            Time vector, same length as data
        out_dir : str
            Directory to save image files
        segment_ms : float
            Segment length in milliseconds, e.g. 400
        overlap : float
            Overlap ratio, e.g. 0.5 means 50% overlap
        f_min, f_max : float
            Frequency range for Stockwell transform
        width : float
            Stockwell width parameter
        decim : int
            Temporal decimation factor
        log_power : bool
            Whether to apply log10 transform
        smooth_sigma : float or None
            Gaussian smoothing sigma for TF image

    Output:
        start_time_dict : dict
            Mapping from filename to segment start time
        stop_time_dict : dict
            Mapping from filename to segment stop time
        """
    data = np.asarray(data).squeeze()
    time = np.asarray(time).squeeze()
    os.makedirs(out_dir, exist_ok=True)

    # Window size: 400 ms
    win_size = int(round(fs * (segment_ms / 1000.0)))
    # 50% overlap, step = win_size * (1 - overlap)
    win_step = int(round(win_size * (1 - overlap)))

    start_time_dict = {}
    stop_time_dict = {}

    counter = 0
    i_start = 0

    while i_start + win_size <= len(data):
        i_stop = i_start + win_size  # Python slice end, exclusive

        segment = data[i_start:i_stop].astype(np.float32)

        img_rgb, freqs = stockwell_transform(segment=segment,fs=fs,fmin=f_min,fmax=f_max,width=width,decim=decim,
                                             log_power=log_power,smooth_sigma=smooth_sigma)

        filename = f"img_{counter:03d}.jpg"
        filepath = os.path.join(out_dir, filename)

        # img_rgb is already in [0,1] RGB format from colormap
        plt.imsave(filepath, img_rgb)

        start_time_dict[filename] = float(time[i_start])
        stop_time_dict[filename] = float(time[i_stop - 1])

        i_start += win_step
        counter += 1

    return start_time_dict, stop_time_dict

class TestImageDataset(Dataset):
    # dataset for test images without labels
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.fnames = sorted(
            [
                os.path.join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.lower().endswith(valid_exts)
            ]
        )

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path


def build_test_dataloader(path, batch_size=8, sz=44):
    # build test dataloader
    # normalization for pretrained ImageNet models
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_ds = TestImageDataset(path, transform=test_tfms)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_ds, test_dl



def compute_CNN(image_path, start_time_dict, stop_time_dict, model_path="../demo_training_saved_model.pth"):
    # using resnet architecture
    # arch = resnet34

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # size of square image in pixels
    sz = 44
    # data: comes from image_path, batch size of 8 for test data, test data located in test folder
    test_ds, test_dl = build_test_dataloader(image_path, batch_size=8, sz=sz)

    # load in pretrained
    state = torch.load(model_path, map_location=device)  # remove map_location parameter if on GPU
    # build PyTorch resnet34 model
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    preds_test = []
    probs_test = []
    test_names = []

    with torch.no_grad():
        for xb, paths in test_dl:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_test.extend(preds.cpu().numpy().tolist())
            probs_test.extend(probs[:, 1].cpu().numpy().tolist())
            test_names.extend([os.path.basename(p) for p in paths])

    test_df = pd.DataFrame(data=test_names, columns=["image_number"])
    test_df["prediction"] = preds_test
    test_df["probability"] = probs_test
    test_df["start time"] = test_df["image_number"].map(start_time_dict)
    test_df["stop time"] = test_df["image_number"].map(stop_time_dict)

    return test_df