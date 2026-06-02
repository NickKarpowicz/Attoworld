"""Data class for handing complex spectra."""

import copy
from dataclasses import dataclass

import numpy as np

from .decorators import yaml_io


@yaml_io
@dataclass(slots=True)
class ComplexSpectrum:
    """Contains a complex spectrum, with spectral weights on a frequency scale.

    Attributes:
        spectrum (np.ndarray): the spectrum in complex128 format
        freq (np.ndarray): the frequencies (float, Hz)

    """

    spectrum: np.ndarray
    freq: np.ndarray

    def lock(self):
        """Make the data immutable."""
        self.spectrum.setflags(write=False)
        self.freq.setflags(write=False)

    def copy(self):
        """Return a copy of the data."""
        return copy.deepcopy(self)

    def to_time_derivative(self):
        r"""Return a ComplexSpectrum corresponding to the time derivative (multiply by $i\omega$)."""
        d_dt = 1j * 2 * np.pi * self.freq * self.spectrum
        return ComplexSpectrum(spectrum=d_dt, freq=np.array(self.freq))

    def to_dazzer_phase(self, multiplier: float = 1.0):
        """Return a two-column numpy array providing the phase in a format suitable for loading as phase.txt by a dazzer.

        Args:
            multiplier: a constant by which to multiply the phase, usually 1 (default) or -1.

        """
        phase = multiplier * np.unwrap(
            np.angle(np.fft.fft(np.fft.fftshift(np.fft.ifft(self.spectrum))))
        )
        phase = phase[1::]
        intensity = np.real(self.spectrum[1::]) ** 2 + np.imag(self.spectrum[1::]) ** 2
        phase_expect = np.sum(phase * intensity) / np.sum(intensity)
        phase -= phase_expect
        wl_nanometers = 2.99792548e17 / self.freq[1::]
        return np.flipud(np.vstack((wl_nanometers, phase)).T)

    def to_bandpassed(self, frequency: float, sigma: float, order: int = 4):
        r"""Return the complex spectrum after applying a supergaussian bandpass filter to the spectrum, of the form
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument.

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian

        """
        new_spectrum = self.spectrum * np.exp(
            -((self.freq - frequency) ** order) / (2 * sigma**order)
        )
        return ComplexSpectrum(spectrum=new_spectrum, freq=np.array(self.freq))
