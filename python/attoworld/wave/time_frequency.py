"""Time-frequency analysis methods for waveforms.

This module provides streamlined time-frequency analysis techniques:
- Short-Time Fourier Transform (STFT)
- Continuous Wavelet Transform (CWT)
- Wigner-Ville Distribution (WVD)

All methods return Spectrogram objects with time and frequency axes in SI units.
"""

from typing import Optional, Tuple

import numpy as np
import pywt
import scipy.signal as sig

from ..data import Spectrogram, Waveform


def stft(
    waveform: Waveform,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    return_complex: bool = False,
) -> Spectrogram:
    """Compute the Short-Time Fourier Transform of a waveform.

    Args:
        waveform: Input Waveform object.
        window: Window function (e.g., 'hann', 'blackman'). Default is 'hann'.
        nperseg: Length of each segment. Defaults to 256 or signal length.
        noverlap: Number of points to overlap. Defaults to nperseg - 1.
        nfft: Length of FFT. Defaults to max(2048, nperseg).
        return_complex: If True, returns complex values. Defaults to False (power).

    Returns:
        Spectrogram: Time-frequency representation.

    """
    uniform_waveform = waveform.to_uniformly_spaced()

    # Defaults
    if nperseg is None:
        nperseg = min(256, len(uniform_waveform.wave))
    if noverlap is None:
        noverlap = nperseg - 1
    if nfft is None:
        nfft = max(2048, nperseg)

    fs = 1.0 / uniform_waveform.dt
    win = sig.get_window(window, nperseg)
    hop = nperseg - noverlap

    # Using ShortTimeFFT for modern interface
    SFT = sig.ShortTimeFFT(
        win,
        hop=hop,
        fs=fs,
        mfft=nfft,
        scale_to="magnitude",
        fft_mode="centered",
        phase_shift=0,
    )

    # Compute STFT
    Zxx = SFT.stft(uniform_waveform.wave)
    f = SFT.f
    t = SFT.t(len(uniform_waveform.wave))

    # Slice to valid range (segments fully within signal)
    # This avoids edge artifacts where the window is partially outside.
    # We calculate the time range corresponding to "valid" convolution.
    t_min = (nperseg / 2) / fs
    t_max = (len(uniform_waveform.wave) - nperseg / 2) / fs

    # Add small tolerance for float comparison
    tol = 0.1 * uniform_waveform.dt
    mask = (t >= t_min - tol) & (t <= t_max + tol)

    Zxx_valid = Zxx[:, mask]
    t_valid = t[mask]

    # Adjust time axis to absolute time
    t_adjusted = t_valid + uniform_waveform.time[0]

    data = Zxx_valid if return_complex else np.abs(Zxx_valid) ** 2

    return Spectrogram(data=data, time=t_adjusted, freq=f)


def cwt(
    waveform: Waveform,
    wavelet: str = "cmor1.5-1.0",
    scales: Optional[np.ndarray] = None,
    num_scales: int = 512,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Spectrogram:
    """Compute the Continuous Wavelet Transform using PyWavelets.

    Args:
        waveform: Input Waveform object.
        wavelet: Name of the wavelet to use (must be a valid PyWavelets name).
            Default is 'cmor1.5-1.0' (Complex Morlet), suitable for optical pulses.
            Format for cmor is 'cmorB-C' where B is bandwidth and C is center frequency.
        scales: Array of scales to use. If None, generated from freq_range or num_scales.
        num_scales: Number of scales to generate if scales is None. Default is 512.
        freq_range: (min_freq, max_freq) in Hz. Used to generate scales if scales is None.

    Returns:
        Spectrogram: Time-frequency representation (power).

    """
    uniform_waveform = waveform.to_uniformly_spaced()
    dt = uniform_waveform.dt
    fs = 1.0 / dt

    # Check if wavelet is available in PyWavelets
    # We rely on pywt to raise error if invalid, but we can try getting center freq first.
    try:
        center_freq = pywt.central_frequency(wavelet)
    except ValueError:
        raise ValueError(
            f"Invalid wavelet name '{wavelet}'. Please use a valid PyWavelets name (e.g., 'cmor1.5-1.0', 'mexh')."
        )

    # Determine scales
    if scales is None:
        if freq_range is not None:
            # f = center_freq * fs / scale  =>  scale = center_freq * fs / f
            # Handle potential division by zero if freq_range includes 0
            f_min = max(freq_range[0], 1e-10 * fs)
            f_max = freq_range[1]

            s_min = center_freq * fs / f_max
            s_max = center_freq * fs / f_min
        else:
            # Default scales: sensible range for optical pulses
            # Smallest scale (highest freq) -> Nyquist limit: f = fs/2 => s = 2*center_freq
            # Largest scale (lowest freq) -> Low freq limit.
            # We choose a range covering typical spectral content.
            # Using existing heuristic:
            s_min = 2 * center_freq  # corresponds to fs/2 approx
            s_max = len(uniform_waveform.wave) / 4  # somewhat arbitrary low freq limit

            # Ensure s_max > s_min
            if s_max <= s_min:
                s_max = s_min * 10

        scales = np.logspace(np.log10(s_min), np.log10(s_max), num_scales)

    # Compute CWT
    cwt_matrix, freqs = pywt.cwt(
        uniform_waveform.wave, scales, wavelet, sampling_period=dt
    )

    # PyWavelets returns frequencies corresponding to scales.
    # If scales are increasing, freqs are decreasing.
    # We sort by frequency ascending.
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    cwt_matrix = cwt_matrix[idx, :]

    # Power spectrogram
    data = np.abs(cwt_matrix) ** 2

    return Spectrogram(data=data, time=uniform_waveform.time, freq=freqs)


def wigner_ville(
    waveform: Waveform,
    nfft: Optional[int] = None,
) -> Spectrogram:
    """Compute the Wigner-Ville Distribution (WVD).
    WVD provides high resolution but contains cross-terms for multi-component signals.

    Args:
        waveform: Input Waveform object.
        nfft: FFT length. Defaults to 2*len(waveform)-1.

    Returns:
        Spectrogram: Real-valued WVD.

    """
    uniform_waveform = waveform.to_uniformly_spaced()
    x = uniform_waveform.wave
    N = len(x)
    dt = uniform_waveform.dt

    if nfft is None:
        nfft = 2 * N - 1

    # WVD Calculation
    # R[n, m] = x[n+m] * x*[n-m]
    # We construct the autocorrelation matrix and FFT it.

    # Optimized implementation logic (same as before but cleaner)
    wvd = np.zeros((nfft, N), dtype=complex)

    # Iterate over time indices
    for n in range(N):
        # tau range: constrained by signal boundaries
        # indices: n +/- m must be in [0, N-1]
        max_lag = min(n, N - 1 - n)

        # Build R vector for this n
        # We can vectorize the lag construction
        m = np.arange(1, max_lag + 1)

        R = np.zeros(nfft, dtype=complex)
        R[0] = x[n] * np.conj(x[n])

        # Positive lags
        val = x[n + m] * np.conj(x[n - m])
        R[m] = val
        # Negative lags (conjugate symmetry)
        R[nfft - m] = np.conj(val)

        wvd[:, n] = np.fft.fft(R)

    wvd_real = np.real(wvd)
    wvd_shifted = np.fft.fftshift(wvd_real, axes=0)

    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=2 * dt))

    return Spectrogram(data=wvd_shifted, time=uniform_waveform.time, freq=freqs)


# USAGE EXAMPLE (commented out):
#
# import numpy as np
# import matplotlib.pyplot as plt
# import attoworld as aw

# # initialize a waveform as a chirped 10-femtosecond Gaussian pulse
# t_fs = np.linspace(-50, 50, 1024) # time in femtoseconds
# f0 = 300.0 / 750.0 # ~0.4 PHz (750nm)
# fwhm = 10  # full-width at half maximum in femtoseconds
# sigma_fs = fwhm / (2 * np.sqrt(np.log(2)))
# chirp = 0.1 # chirp in 1/fs^2

# envelope = np.exp(-t_fs**2 / (2 * sigma_fs**2) + 1j * (chirp * t_fs)**2)
# waveform = aw.data.ComplexEnvelope(
#     envelope=envelope,
#     time=t_fs*1e-15,
#     dt=(t_fs[1]-t_fs[0])*1e-15,
#     carrier_frequency=f0*1e+15
# ).to_waveform()

# # Short-time Fourier transform
# nperseg=512
# spec_stft = aw.wave.stft(waveform)

# # Continuous wavelet transform
# spec_cwt = aw.wave.cwt(waveform)

# # Wigner-Ville transform
# spec_wvd = aw.wave.wigner_ville(waveform)

# # plot the spectrograms
# methods = [
#     (spec_stft, "Short-time Fourier transform", True),
#     (spec_cwt, "Continuous wavelet transform", True),
#     (spec_wvd, "Wigner-Ville transform", False),
# ]
# for spectrogram, title, take_sqrt in methods:
#     fig, ax = plt.subplots()
#     ax.set_xlim(-20., 20.)  # fs
#     ax.set_ylim(150., 650.) # THz
#     ax.set_title(title)
#     spectrogram.plot(ax, take_sqrt=take_sqrt)

# plt.show()
