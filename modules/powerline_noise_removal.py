import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy import signal
from scipy.signal import filtfilt, freqz, iirnotch


def ft_preproc_dftfilter(
    dat, Fs, Fl=50, dftreplace="neighbour", dftbandwidth=None, dftneighbourwidth=None
):
    """
    Python version of FieldTrip's ft_preproc_dftfilter.

    FT_PREPROC_DFTFILTER reduces power line noise (50 or 60Hz) via two
    alternative methods:
    A) DFT filter (Flreplace = 'zero') or
    B) Spectrum Interpolation (Flreplace = 'neighbour').

    A) The DFT filter applies a notch filter to the data to remove the 50Hz
    or 60Hz line noise components ('zeroing'). This is done by fitting a sine
    and cosine at the specified frequency to the data and subsequently
    subtracting the estimated components. The longer the data is, the sharper
    the spectral notch will be that is removed from the data.
    Preferably the data should have a length that is a multiple of the
    oscillation period of the line noise (i.e. 20ms for 50Hz noise). If the
    data is of different lenght, then only the first N complete periods are
    used to estimate the line noise. The estimate is subtracted from the
    complete data.

    B) Alternatively line noise is reduced via spectrum interpolation
    (Leske & Dalal, 2019, NeuroImage 189,
    doi: 10.1016/j.neuroimage.2019.01.026)
    The signal is:
    I)   transformed into the frequency domain via a discrete Fourier
        transform (DFT),
    II)  the line noise component (e.g. 50Hz, Flwidth = 1 (±1Hz): 49-51Hz) is
        interpolated in the amplitude spectrum by replacing the amplitude
           of this frequency bin by the mean of the adjacent frequency bins
           ('neighbours', e.g. 49Hz and 51Hz).
           Neighwidth defines frequencies considered for the mean (e.g.
           Neighwidth = 2 (±2Hz) implies 47-49 Hz and 51-53 Hz).
           The original phase information of the noise frequency bin is
           retained.
     III) the signal is transformed back into the time domain via inverse DFT
           (iDFT).
     If Fline is a vector (e.g. [50 100 150]), harmonics are also considered.
     Preferably the data should be continuous or consist of long data segments
     (several seconds) to avoid edge effects. If the sampling rate and the
     data length are such, that a full cycle of the line noise and the harmonics
     fit in the data and if the line noise is stationary (e.g. no variations
     in amplitude or frequency), then spectrum interpolation can also be
     applied to short trials. But it should be used with caution and checked
     for edge effects.

     Use as
       [filt] = ft_preproc_dftfilter(dat, Fsample, Fline, varargin)
     where
       dat             data matrix (Nchans X Ntime)
       Fsample         sampling frequency in Hz
       Fline           line noise frequency (and harmonics)

     Additional input arguments come as key-value pairs:

       Flreplace       'zero' or 'neighbour', method used to reduce line noise, 'zero' implies DFT filter, 'neighbour' implies spectrum interpolation
       dftbandwidth        bandwidth of line noise frequencies, applies to spectrum interpolation, in Hz
       dftneighbourwidth     width of frequencies neighbouring line noise frequencies, applies to spectrum interpolation (Flreplace = 'neighbour'), in Hz

     The line frequency should be specified as a single number for the DFT filter.
     If omitted, a European default of 50Hz will be assumed

     If the data contains NaNs, the output of the affected channel(s) will be
     all(NaN).

     See also PREPROC
     Undocumented option:
       Fline can be a vector, in which case the regression is done for all
       frequencies in a single shot. Prerequisite is that the requested
       frequencies all fit with an integer number of cycles in the data.

     Copyright (C) 2003, Pascal Fries
     Copyright (C) 2003-2015, Robert Oostenveld
     Copyright (C) 2016, Sabine Leske

    This file is part of FieldTrip, see http://www.fieldtriptoolbox.org
    for the documentation and details.

        FieldTrip is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        FieldTrip is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with FieldTrip. If not, see <http://www.gnu.org/licenses/>.

    Matlab code: https://github.com/Frederik-D-Weber/sleeptrip/blob/15662f826e9c03a1d21c7d5d09e7c61d456e8b71/preproc/ft_preproc_dftfilter.m#L151

    """

    dat = np.asarray(dat)
    n_channels, n_samples = dat.shape
    Fl = np.atleast_1d(Fl)

    if np.isnan(dat).any():
        print("Warning: Input data contains NaNs.")

    if dftreplace == "zero":
        time = np.arange(n_samples) / Fs
        filt = dat.copy()

        for freq in Fl:
            # calculate the number of cycles for freq in n_samples
            n_cycles = int(np.floor(n_samples * freq / Fs))
            n_samples_fit = int(n_cycles * Fs / freq)

            if n_samples_fit < 1 or n_samples_fit > n_samples:
                print(f"Skipping {freq} Hz: not enough cycles in signal.")
                continue

            sel = slice(0, n_samples_fit)
            time_fit = time[sel]

            # Remove mean temporarily
            mean_vals = np.nanmean(filt[:, sel], axis=1, keepdims=True)
            dat_centered = filt - mean_vals

            # Build complex sine basis
            basis = np.exp(1j * 2 * np.pi * freq * time)
            # e^(j*2πft) = cos(2πft) + j*sin(2πft)
            basis_fit = basis[sel]

            # Estimate complex amplitude for each channel
            ampl = 2 * (dat_centered[:, sel] @ np.conj(basis_fit)) / n_samples_fit

            # Reconstruct estimated sinusoid for full time series
            est = np.outer(ampl, basis)

            # Subtract and restore mean
            filt = np.real(dat_centered - est) + mean_vals

        return filt

    elif dftreplace == "neighbour":
        Flwidth = np.atleast_1d(dftbandwidth)
        Neighwidth = np.atleast_1d(dftneighbourwidth)

        if len(Fl) != len(Flwidth) or len(Fl) != len(Neighwidth):
            raise ValueError(
                "Fl, dftbandwidth, and dftneighbourwidth must have the same length."
            )

        freqs = fftfreq(n_samples, d=1.0 / Fs)
        data_fft = fft(dat, axis=1)

        for i, freq in enumerate(Fl):
            fw = Flwidth[i]
            nw = Neighwidth[i]

            # Define interpolation range
            f2int = [freq - fw, freq + fw]
            f4int = [f2int[0] - nw, f2int[0], f2int[1], f2int[1] + nw]

            # Indices of frequencies
            smpl2int = np.where((freqs >= f2int[0]) & (freqs <= f2int[1]))[0]
            smpl4int = np.where(
                ((freqs >= f4int[0]) & (freqs < f4int[1]))
                | ((freqs > f4int[2]) & (freqs <= f4int[3]))
            )[0]

            # Compute new amplitude from neighbors
            amp_neighbors = np.mean(
                np.abs(data_fft[:, smpl4int]), axis=1, keepdims=True
            )
            phase_orig = np.angle(data_fft[:, smpl2int])

            # Replace amplitude, preserve phase
            data_fft[:, smpl2int] = amp_neighbors * np.exp(1j * phase_orig)

            # Inverse FFT to time domain
        filt = np.real(ifft(data_fft, axis=1))

        return filt

    else:
        raise ValueError(f"Unknown dftreplace method: {dftreplace}")


def ft_preproc_notch(dat, fs, powerline_freqs=[50], q=30, plot_response=False):
    """
    Apply notch filters to remove powerline noise and harmonics (e.g., 50Hz, 100Hz, 150Hz, ...).

    Parameters:
        dat (np.ndarray): Input signal (1D or 2D: channels x time).
        fs (float): Sampling frequency.
        powerline_freqs (list or np.ndarray): List of powerline frequencies to remove (e.g., [50, 100, 150]).
        q (float): Quality factor (default=30).
        plot_response (bool): If True, plot frequency and phase response for each notch filter.

    Returns:
        np.ndarray: Filtered signal.
    """

    notch_filt_data = dat.copy()

    for f_notch in powerline_freqs:
        if f_notch >= fs / 2:
            print(f"Skipping {f_notch} Hz (above Nyquist)")
            continue

        # Design notch filter
        b_notch, a_notch = iirnotch(f_notch, q, fs)

        # Optional: plot frequency and phase response
        if plot_response:
            w, h = freqz(b_notch, a_notch, fs=fs)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(w, 20 * np.log10(abs(h)), label=f"{f_notch} Hz")
            plt.axvline(f_notch, color="r", linestyle="--")
            plt.title(f"Magnitude Response - Notch @ {f_notch} Hz")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(w, np.unwrap(np.angle(h)), label=f"{f_notch} Hz")
            plt.axvline(f_notch, color="r", linestyle="--")
            plt.title(f"Phase Response - Notch @ {f_notch} Hz")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase (radians)")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Apply filter
        if notch_filt_data.ndim == 1:
            notch_filt_data = filtfilt(b_notch, a_notch, notch_filt_data)
        else:
            notch_filt_data = np.vstack(
                [filtfilt(b_notch, a_notch, ch) for ch in notch_filt_data]
            )

    return notch_filt_data


def ft_preproc_adaptivefilter(dat, fs, powerline_freq=50, mu=0.001):
    """
    Adaptive notch filter based on the LMS algorithm.
    Supports single-channel (1D) or multi-channel (2D: channels x time) input.

    Parameters:
        dat : ndarray
            Input signal (1D or 2D array).
        fs : float
            Sampling frequency (Hz).
        powerline_freq : float
            Target notch frequency, e.g., 50.
        mu : float
            Adaptation step size (learning rate).

    Returns:
        filtered_dat : ndarray
            Signal after adaptive notch filtering (same shape as input).
    """

    # Validate parameters
    if mu <= 0 or mu > 0.1:
        raise ValueError("Learning rate (mu) should be in (0, 0.1] range")

    # Handle 1D input by reshaping to 2D
    is_1d = dat.ndim == 1
    if is_1d:
        dat = dat[np.newaxis, :]  # Shape becomes (1, time)

    n_channels, n_samples = dat.shape
    filtered_dat = np.copy(dat)

    for ch in range(n_channels):
        signal = dat[ch]
        filtered_signal = np.zeros(n_samples)
        estimated_noise = np.zeros(n_samples)
        weights = np.zeros(2)  # [sin component, cos component]

        for k in range(n_samples):
            # Generate reference sinusoidal components
            ref = np.array(
                [
                    np.sin(2 * np.pi * powerline_freq * k / fs),
                    np.cos(2 * np.pi * powerline_freq * k / fs),
                ]
            )

            # Estimate the noise (harmonic)
            estimated_noise[k] = np.dot(weights, ref)

            # Compute the residual (cleaned signal)
            filtered_signal[k] = signal[k] - estimated_noise[k]

            # Update filter weights using LMS rule
            weights += 2 * mu * filtered_signal[k] * ref

        filtered_dat[ch] = filtered_signal

        # Return to original shape if input was 1D
    if is_1d:
        return filtered_dat[0]
    return filtered_dat


def ft_preproc_adaptivefilter_rls(dat, fs, powerline_freq=50, lambda_=0.998, delta=0.5):
    """
    Adaptive notch filter using RLS (Recursive Least Squares).
    Supports single-channel (1D) or multi-channel (2D: channels x time) input.

    Parameters:
        dat : ndarray
            Input signal (1D or 2D array).
        fs : float
            Sampling frequency (Hz).
        powerline_freq : float
            Target notch frequency to remove (e.g., 50 Hz).
        lambda_ : float
            Forgetting factor for RLS (0 < lambda_ <= 1), controls adaptation speed.
        delta : float
            Initial regularization term for inverse correlation matrix P.

    Returns:
        filtered_dat : ndarray
            Signal after adaptive RLS notch filtering (same shape as input).
    """
    if lambda_ <= 0 or lambda_ > 1:
        raise ValueError("Forgetting factor lambda_ must be in (0, 1]")
    if delta <= 0:
        raise ValueError("Delta must be positive")

    is_1d = dat.ndim == 1
    if is_1d:
        dat = dat[np.newaxis, :]

    n_channels, n_samples = dat.shape
    filtered_dat = np.copy(dat)

    for ch in range(n_channels):
        signal = dat[ch]
        # Time vector
        t = np.arange(n_samples) / fs

        # Reference signals: sin & cos of target frequency
        X = np.vstack(
            [
                np.sin(2 * np.pi * powerline_freq * t),
                np.cos(2 * np.pi * powerline_freq * t),
            ]
        ).T  # shape: (n_samples, 2)

        # RLS initialization
        w = np.zeros(2)  # filter weights [sin, cos]
        P = np.eye(2) / delta  # inverse correlation matrix
        eps = 1e-10  # small regularization term
        filtered_signal = np.zeros(n_samples)

        for k in range(n_samples):
            xk = X[k].reshape(2, 1)
            yk = signal[k]

            y_hat = float(xk.T @ w.reshape(2, 1))
            ek = yk - y_hat

            # Gain vector with regularization
            denominator = lambda_ + xk.T @ P @ xk + eps
            gk = (P @ xk) / denominator

            # Update weights and inverse correlation matrix
            w += gk.flatten() * ek
            P = (P - gk @ xk.T @ P) / lambda_

            # Store filtered sample
            filtered_signal[k] = ek  # y - estimated_noise

        filtered_dat[ch] = filtered_signal

    # Return to original 1D shape if input was 1D
    if is_1d:
        return filtered_dat[0]
    else:
        return filtered_dat


def calculate_parr(
    original_signal, filtered_signal, fs, powerline_freq=50, harmonic_n=9, bandwidth=2.0
):
    """
    Calculate Power Attenuation Ratio of Powerline and Harmonics (PARR) in dB.

    Parameters:
        original_signal (array): Original noisy signal (1D array).
        filtered_signal (array): Filtered signal (1D array).
        fs (float): Sampling frequency in Hz.
        powerline_freq (float): Fundamental powerline frequency (e.g., 50 or 60 Hz).
        harmonic_n (int): Number of harmonics to include.
        bandwidth (float): Bandwidth around each harmonic to consider (Hz).

    Returns:
        parr (float): Power attenuation ratio in dB.
        harmonic_freqs (list): Frequencies used for calculation.
    """

    # Calculate PSD using Welch's method
    f, Pxx_orig = signal.welch(original_signal, fs, nperseg=1024)
    f, Pxx_filt = signal.welch(filtered_signal, fs, nperseg=1024)

    # Identify harmonic frequencies within Nyquist range
    harmonic_freqs = []
    for n in range(1, harmonic_n + 1):
        harmonic = n * powerline_freq
        if harmonic < fs / 2:  # Nyquist criterion
            harmonic_freqs.append(harmonic)

    # Calculate power in harmonic bands
    power_orig = 0
    power_filt = 0
    for freq in harmonic_freqs:
        mask = (f >= freq - bandwidth / 2) & (f <= freq + bandwidth / 2)
        if np.any(mask):
            power_orig += np.trapz(Pxx_orig[mask], f[mask])
            power_filt += np.trapz(Pxx_filt[mask], f[mask])

    # Avoid division by zero
    if power_filt < 1e-10:
        power_filt = 1e-10

    parr = 10 * np.log10(power_orig / power_filt)
    return parr, harmonic_freqs


def calculate_ppr_specific_band(
    original_signal,
    filtered_signal,
    fs,
    target_band=(100, 300),
    exclude_harmonics=None,
    exclude_bw=None,
):
    """
    Calculate Power Preservation Ratio (PPR) for a specific frequency band while
    excluding harmonics within that band.

    Parameters:
        original_signal (array): Original noisy signal.
        filtered_signal (array): Filtered signal.
        fs (float): Sampling frequency.
        target_band (tuple): Frequency range of interest (low, high) in Hz.
        exclude_harmonics (list): List of harmonic frequencies to exclude.
        exclude_bw (float): Bandwidth around harmonics to exclude (Hz).

    Returns:
        ppr (float): Power preservation ratio in dB.
        effective_band (list): Actual frequency ranges used after harmonic exclusion.
    """

    # Calculate PSD
    f, Pxx_orig = signal.welch(original_signal, fs, nperseg=1024)
    f, Pxx_filt = signal.welch(filtered_signal, fs, nperseg=1024)

    # Create mask for target band
    mask = (f >= target_band[0]) & (f <= target_band[1])

    # Exclude harmonics if specified
    if exclude_harmonics is not None and exclude_bw is not None:
        for harmonic in exclude_harmonics:
            if target_band[0] <= harmonic <= target_band[1]:
                mask &= ~(
                    (f >= harmonic - exclude_bw / 2) & (f <= harmonic + exclude_bw / 2)
                )

    # Calculate power in the effective band
    power_orig = np.trapz(Pxx_orig[mask], f[mask])
    power_filt = np.trapz(Pxx_filt[mask], f[mask])

    # Handle edge cases
    if power_orig < 1e-10:
        return 0.0, []

    ppr = 10 * np.log10(power_filt / power_orig)

    # Generate effective frequency ranges
    effective_band = []
    in_band = False
    start_freq = None
    mask_indices = np.where(mask)[0]
    for i in range(len(mask_indices)):
        idx = mask_indices[i]
        freq = f[idx]
        if not in_band:
            start_freq = freq
            in_band = True
        if i == len(mask_indices) - 1 or mask_indices[i + 1] != idx + 1:
            effective_band.append((start_freq, freq))
            in_band = False

    return ppr, effective_band


@numba.njit
def _count_matches(signal, m, r):
    N = len(signal)
    count = 0
    for i in range(N - m):
        for j in range(i + 1, N - m):
            match = True
            for k in range(m):
                if abs(signal[i + k] - signal[j + k]) > r:
                    match = False
                    break
            if match:
                count += 1
    return count


def calculate_sampen_fast(signal, m=2, r=0.2):
    """
    Calculate Sample Entropy (SampEn) of a time series.

    Sample Entropy is a measure of time series regularity that is similar to approximate entropy
    but has better consistency over varying signal lengths.

    Parameters:
        signal (array): Input time series (1D array).
        m (int): Template length (typically 2).
        r (float): Tolerance threshold (typically 0.2*std).

    Returns:
        sampen (float): Sample entropy value.
    """

    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if N <= m:
        return 0.0

    std = np.std(signal)
    if std < 1e-10:
        return 0.0
    r_scaled = r * std

    B = _count_matches(signal, m, r_scaled)
    A = _count_matches(signal, m + 1, r_scaled)

    if B == 0 or A == 0:
        return np.inf
    return -np.log(A / B)


def calculate_sampen_avg(signal, window_size=10000, step=1000, m=2, r=0.2):
    """
    Calculate the average Sample Entropy (SampEn) of a time series.

    Sample Entropy is a measure of time series regularity that is similar to approximate entropy
    but has better consistency over varying signal lengths.

    Parameters:
        signal (array): Input time series (1D array).
        m (int): Template length (typically 2).
        r (float): Tolerance threshold (typically 0.2*std).

    Returns:
        sampen (float): Sample entropy value.
    """

    sampen_values = []
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size]
        se = calculate_sampen_fast(window, m, r)
        sampen_values.append(se)
    sampen_values = np.array(sampen_values)
    avg_sampen = np.mean(sampen_values[np.isfinite(sampen_values)])  # 排除 inf
    return avg_sampen


"""
def compare_psd(raw_signal, filtered_signal, fs, Powerline_freq = [50], nperseg=2048):

    f_raw, psd_raw = welch(raw_signal, fs, nperseg=nperseg)
    f_filt, psd_filt = welch(filtered_signal, fs, nperseg=nperseg)

    def band_power(f, psd, center, width=1.0):
        band = (f >= center - width/2) & (f <= center + width/2)
        return np.sum(psd[band])

    power_ratios = {}
    for freq in Powerline_freq:
        power_raw = band_power(f_raw, psd_raw, freq)
        power_filt = band_power(f_filt, psd_filt, freq)
        reduction_db = 10 * np.log10(power_raw / (power_filt + 1e-12))
        power_ratios[f"{freq}Hz_reduction_dB"] = reduction_db

    return power_ratios


def shape_similarity(x_raw, x_filtered):
    cc = np.corrcoef(x_raw, x_filtered)[0, 1]
    return cc

"""
