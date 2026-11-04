"""Data class for handing complex spectra."""

import copy
from dataclasses import dataclass

import numpy as np
import scipy.constants as constants
import scipy.optimize as opt

from ..numeric import derivative
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

    def to_applied_phase_series(
        self,
        l0: float = 800.0,
        sign: int = 1,
        delay: float = 0.0,
        gdd: float = 0.0,
        tod: float = 0.0,
        fod: float = 0.0,
        ):
        """Return a ComplexSpectrum with a given parameterized phase applied.

        Args:
            l0: the central wavelength of the expansion (nm)
            sign: multiplier (e.g. -1 changes sign)
            delay: time delay (fs)
            gdd: group delay dispersion (fs^2)
            tod: third order dispersion (fs^3)
            fod: fourth order dispersion (fs^4)

        """
        f0 = 1e9*constants.speed_of_light/l0
        w = 2.0 * np.pi * (self.freq-f0)
        phase = 2.0e-15*np.pi*self.freq * delay + 0.5e-30 * gdd * (w**2) + (tod/6.0e45) * (w**3) + (fod/24.0e60) * (w**4)
        return ComplexSpectrum(spectrum=np.exp(1j * sign * phase) * self.spectrum, freq=np.array(self.freq))

    def get_optimized_phase_series(
        self, l0: float = 800.0,
        sign: int = 1,
        max_gdd: float = 0.0,
        max_tod: float = 0.0,
        max_fod: float = 0.0
    ):
        """Optimize phase series (delay, GDD, TOD, FOD) for peak intensity."""
        def get_inverse_intensity(x):
            gdd, tod, fod = x
            new_pulse = np.fft.ifft(self.to_applied_phase_series(l0, sign, 0.0, gdd, tod, fod).spectrum, 16*len(self.spectrum))
            return 1.0/np.max(np.abs(new_pulse))


        bounds = [
            (-max_gdd, max_gdd),
            (-max_tod, max_tod),
            (-max_fod, max_fod)
        ]

        res = opt.differential_evolution(
            get_inverse_intensity,
            bounds,
            popsize=32,
            maxiter=16384,
            tol = 1e-4
        )

        return res.x
    def to_dazzer_phase(
        self,
        multiplier: float = 1.0,
        filter_width: float = 100e-15,
        filter_order: int = 4,
    ):
        """Return a two-column numpy array providing the phase in a format suitable for loading as phase.txt by a dazzer.

        Args:
            multiplier: a constant by which to multiply the phase, usually 1 (default) or -1.
            filter_width: time constant for the phase filter in seconds (default 100e-15)
            filter_order: order of the supergaussian time filter (default 4)

        """
        phase = multiplier * np.unwrap(
            np.angle(np.fft.fft(np.fft.fftshift(np.fft.ifft(self.spectrum))))
        )
        phase = phase[1::]
        intensity = np.real(self.spectrum[1::]) ** 2 + np.imag(self.spectrum[1::]) ** 2
        f0 = np.sum(self.freq[1::] * intensity) / np.sum(intensity)

        wl_nanometers = 2.99792548e17 / self.freq[1::]

        def apply_time_filter(
            phase, frequencies, zero_freq, filter_width, filter_order
        ):
            freq_step = np.abs(frequencies[1] - frequencies[0])
            zero_index = (np.abs(frequencies - zero_freq)).argmin()
            dphase = derivative(np.array(phase), 1, neighbors=3)
            t = np.fft.fftfreq(len(phase), d=freq_step)
            sg_filter = np.exp(-(t**filter_order) / (2 * filter_width**filter_order))
            filtered = np.cumsum(np.real(np.fft.ifft(sg_filter * np.fft.fft(dphase))))
            filtered -= filtered[zero_index]
            return filtered

        filtered_phase = apply_time_filter(
            phase, self.freq[1::], f0, filter_width, filter_order
        )
        return np.flipud(np.vstack((wl_nanometers, filtered_phase)).T)

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
