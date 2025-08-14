"""Define the base version of the Waveform class, which is extended in interop.py."""

import copy
from dataclasses import dataclass

import numpy as np
import scipy.signal as sig

from ..numeric import (
    find_maximum_location,
    fwhm,
    interpolate,
)
from .decorators import yaml_io


@yaml_io
@dataclass(slots=True)
class Waveform:
    """Contains data describing an electric field waveform.
    In SI units.

    Attributes:
        wave (np.ndarray): the waveform data
        time (np.ndarray): time vector corresponding to the wave array
        dt (float): the time step of time, if it is uniformly spaced
        is_uniformly_spaced (bool): says whether or not the time steps are uniform

    """

    wave: np.ndarray
    time: np.ndarray
    dt: float
    is_uniformly_spaced: bool = False

    def lock(self):
        """Make the data immutable."""
        self.wave.setflags(write=False)
        self.time.setflags(write=False)

    def copy(self):
        """Returns a deep copy."""
        return copy.deepcopy(self)

    def time_fs(self):
        """Returns the time in femtoseconds."""
        if self.time is not None:
            return 1e15 * self.time
        raise Exception("No data")

    def to_uniformly_spaced(self):
        """Returns a version of the struct with uniformly spaced time."""
        if self.is_uniformly_spaced:
            return self
        timesteps = np.abs(np.diff(self.time))
        new_dt = np.min(timesteps[timesteps > 0.0])
        new_time_length = self.time[-1] - self.time[0]
        new_time = self.time[0] + new_dt * np.array(
            range(int(new_time_length / new_dt))
        )
        return Waveform(
            wave=interpolate(new_time, self.time, self.wave),
            time=new_time,
            dt=new_dt,
            is_uniformly_spaced=True,
        )

    def to_windowed(self, window_desc):
        """Create a windowed version of the waveform. Output will be uniformly spaced in time, even if current state isn't.

        Args:
            window_desc: String or tuple describing the desired window in the same format as scipy.signal.windows.get_window.

        Examples:
            >>> waveform_with_tukey_window = waveform.to_windowed('tukey')
            >>> waveform_with_supergaussian_window = waveform.to_windowed(('general_gaussian', 2, 500))

        """
        uniform_self = self.to_uniformly_spaced()
        new_wave = uniform_self.wave * sig.windows.get_window(
            window_desc, Nx=uniform_self.wave.shape[0]
        )
        return Waveform(
            wave=new_wave,
            time=uniform_self.time,
            dt=uniform_self.dt,
            is_uniformly_spaced=True,
        )

    def to_bandpassed(self, frequency: float, sigma: float, order: int = 4):
        r"""Apply a bandpass filter, with the same spec as to_bandpassed method of ComplexSpectrum:
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument.

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian

        """
        return (
            self.to_complex_spectrum()
            .to_bandpassed(frequency, sigma, order)
            .to_waveform()
        )

    def to_normalized(self):
        """Return a normalized version of the waveform."""
        max_loc, max_val = find_maximum_location(
            np.abs(np.array(sig.hilbert(self.wave)))
        )
        return Waveform(
            wave=self.wave / max_val,
            time=np.array(self.time),
            dt=self.dt,
            is_uniformly_spaced=self.is_uniformly_spaced,
        )

    def get_envelope_fwhm(self) -> float:
        """Get the full-width-at-half-maximum of the intensity envelope.

        Returns:
            float: the FWHM

        """
        return self.to_complex_envelope().get_fwhm()

    def get_field_squared_fwhm(self):
        """Get the full-width-at-half-maximum of the square of the field.

        Returns:
            float: the FWHM

        """
        uniform_self = self.to_uniformly_spaced()
        return fwhm(uniform_self.wave**2, uniform_self.dt)
