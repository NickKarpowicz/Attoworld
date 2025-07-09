"""Tools for working with spectra."""

from typing import Optional

import numpy as np
from scipy import constants

from ..numeric import interpolate


def frequency_to_wavelength(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
):
    """Convert a frequency spectrum in W/Hz into a wavelength spectrum in W/m. SI units.

    Args:
        frequencies (np.ndarray): the frequencies included in the data, in Hz
        spectrum (np.ndarray): the spectrum corresponding to the input frequency scale
        wavelengths: (optional) wavelength vector for the output. If not specified, a the output data will be on the same grid, but scaled
    Returns:
        wavelengths: wavelengths in m
        scaled_spectrum: the wavelength-domain spectrum

    """
    if wavelengths is None:
        wavelengths = constants.speed_of_light / frequencies[frequencies > 0.0]
        spectrum = spectrum[frequencies > 0.0]
    else:
        wavelengths = wavelengths[wavelengths > 0.0]

    return wavelengths, spectrum / (wavelengths**2)


def wavelength_to_frequency(
    wavelengths_nm: np.ndarray,
    spectrum: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
):
    """Convert a wavelength spectrum in W/nm into a frequency spectrum in W/THz.

    Args:
        wavelengths_nm (np.ndarray): the wavelengths included in the data, in nanometers
        spectrum (np.ndarray): the spectrum corresponding to the input wavelength scale
        frequencies: (optional) frequency vector for the output. If not specified, a vector will be calculated such that resolution and range are preserved.

    Returns:
        f: frequencies (Hz)
        scaled_spectrum: the frequency-domain spectrum

    """
    # Contributed by Nick Karpowicz
    input_frequencies = 1e9 * constants.speed_of_light / wavelengths_nm

    if frequencies is None:
        frequency_step = np.min(np.abs(np.diff(input_frequencies)))
        min_frequency = np.min(input_frequencies)
        max_frequency = np.max(input_frequencies)
        frequency_count = int(np.ceil(max_frequency - min_frequency) / frequency_step)
        frequencies = min_frequency + frequency_step * np.array(
            range(frequency_count), dtype=float
        )

    # apply Jakobian scaling and use W/THz
    scaled_spectrum = (
        1e12 * constants.speed_of_light * spectrum / (input_frequencies**2)
    )

    scaled_spectrum = interpolate(
        frequencies, input_frequencies, scaled_spectrum, inputs_are_sorted=False
    )

    return frequencies, scaled_spectrum


def transform_limited_pulse_from_spectrometer(
    wavelengths_nm: np.ndarray, spectrum: np.ndarray, gate_level: Optional[float] = None
):
    """Calculate the transform-limited pulse corresponding to a spectrum.

    Args:
        wavelengths_nm: the wavelengths included in the data, in nanometers
        spectrum: the spectrum corresponding to the input wavelength scale
        gate_level: (optional) level, relative to the maximum at which to apply a gate to the spectrum. For example, with gate_level=0.01, values less than 1% of the maximum signal will be set to zero

    Returns:
        t: time vector (s)
        pulse: the pulse intensity vs. time

    """
    f, spec = wavelength_to_frequency(wavelengths_nm, spectrum)
    if f is None:
        raise ValueError("No frequencies!")
    df = f[1] - f[0]
    t = np.fft.fftshift(np.fft.fftfreq(f.shape[0], d=df))
    gated_spectrum = np.array(spec)
    if gate_level is not None:
        gated_spectrum[spec < (gate_level * np.max(spec))] = 0.0

    pulse = np.fft.fftshift(np.abs(np.fft.ifft(np.sqrt(gated_spectrum)))) ** 2

    return t, pulse
