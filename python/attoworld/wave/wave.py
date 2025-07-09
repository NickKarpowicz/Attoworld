"""Waveform processing tools."""

import numpy as np
import scipy.optimize as opt
import scipy.signal as sig


def align_waves(
    waves, dt: float, frequency_roi_start: float, frequency_roi_stop: float
):
    """Align a set of waveforms, inside of a 2D numpy array.

    Args:
        waves: set of waveforms
        dt (float): time step (assuming constant spacing), (s)
        frequency_roi_start (float): start frequency of region of interest for alignment (Hz)
        frequency_roi_stop (float): stop frequency of region of interest for alignment (Hz)

    Returns:
        np.ndarray: set of aligned waves

    """
    waves_f = np.array(waves)

    # windowing and offset removal
    for i in range(waves_f.shape[0]):
        waves_f[i, :] = sig.windows.tukey(waves_f.shape[1]) * (
            waves_f[i, :] - np.mean(waves_f[i, :])
        )

    # find frequency roi
    f = np.fft.fftfreq(waves.shape[1], dt)
    f0 = np.argmin(np.abs(f - frequency_roi_start))
    f1 = np.argmin(np.abs(f - frequency_roi_stop))
    w_roi = 2 * np.pi * f[f0:f1]

    # get active spectral region
    waves_f = np.fft.fft(waves, axis=1)
    waves_roi = np.array(waves_f[:, f0:f1])
    waves_roi /= np.max(np.max(np.abs(waves_roi)))

    # apply tau phase shifts
    def apply_taus(spectrum, taus, w):
        spectrum_shifted = np.zeros(spectrum.shape, dtype=complex)
        for i in range(spectrum.shape[0]):
            spectrum_shifted[i, :] = np.exp(-1j * 1e-18 * taus[i] * w) * spectrum[i, :]
        return spectrum_shifted

    # return fitting weights
    def get_residual(taus):
        shifted = apply_taus(waves_roi, taus, w_roi)
        mean_amplitudes = np.mean(shifted, axis=0)
        return 5.0 - np.abs(mean_amplitudes)

    # apply fitting
    res = opt.least_squares(
        get_residual, np.zeros(waves.shape[0]), ftol=1e-12, max_nfev=16384
    )

    # remove mean shift
    res.x -= np.mean(res.x)

    print(f"Rms shift in attoseconds: {np.std(res.x)}")
    return np.real(np.fft.ifft(apply_taus(waves_f, res.x, 2 * np.pi * f)))
