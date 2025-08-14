"""Base class for containing spectral data."""

import copy
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import constants

from ..numeric import (
    derivative,
    interpolate,
)
from .decorators import yaml_io


@yaml_io
@dataclass(slots=True)
class IntensitySpectrum:
    """Contains an intensity spectrum - real valued. SI units.

    Attributes:
        spectrum (Optional[np.ndarray]): the spectral intensity
        phase (Optional[np.ndarray]): the spectral phase
        freq (Optional[np.ndarray]): the frequencies corresponding to the spectrum
        wavelength (Optional[np.ndarray]): the wavelengths
        is_frequency_scaled (bool): has the lamda^-2 Jakobian been applied to Fourier transformed data?

    """

    spectrum: np.ndarray
    phase: Optional[np.ndarray]
    freq: np.ndarray
    wavelength: np.ndarray
    is_frequency_scaled: bool = False

    def lock(self):
        """Make the data immutable."""
        self.spectrum.setflags(write=False)
        self.freq.setflags(write=False)
        self.wavelength.setflags(write=False)
        if self.phase is not None:
            self.phase.setflags(write=False)

    def copy(self):
        """Make a copy of the class."""
        return copy.deepcopy(self)

    def wavelength_nm(self):
        """Returns:
        Optional[np.ndarray]: the wavelengths in nm.

        """
        if self.wavelength is not None:
            return 1e9 * self.wavelength
        return None

    def wavelength_micron(self):
        """Returns:
        Optional[np.ndarray]: the wavelengths in microns.

        """
        if self.wavelength is not None:
            return 1e6 * self.wavelength
        return None

    @staticmethod
    def from_spectrometer_spectrum_nanometers(
        wavelengths_nanometers: np.ndarray, spectrum: np.ndarray
    ):
        """Generate an instance based on a spepctrum array and wavelength array in nm (typical spectrometer data).

        Args:
            wavelengths_nanometers: the wavelengths in nanometers
            spectrum: the spectral intensity

        """
        wavelengths = 1e-9 * wavelengths_nanometers
        freqs = constants.speed_of_light / wavelengths
        phase = np.zeros(freqs.shape, dtype=float)
        return IntensitySpectrum(
            spectrum=spectrum, wavelength=wavelengths, freq=freqs, phase=phase
        )

    def get_wavelength_spectrum(self):
        """Return a wavelength-scaled spectrum, independent of the state of the current instance.

        Returns:
            np.ndarray, np.ndarray: wavelength and spectrum

        """
        if self.is_frequency_scaled:
            if self.freq is not None and self.spectrum is not None:
                from ..spectrum import frequency_to_wavelength

                return frequency_to_wavelength(self.freq, self.spectrum)
            raise Exception("Missing data")
        if self.wavelength is not None and self.spectrum is not None:
            return self.wavelength, self.spectrum
        raise Exception("Missing data")

    def get_frequency_spectrum(self):
        """Return a frequency-scaled spectrum, independent of the state of the current instance.

        Returns:
            np.ndarray, np.ndarray: wavelength and spectrum

        """
        if self.is_frequency_scaled:
            if self.freq is not None and self.spectrum is not None:
                return self.freq, self.spectrum
            raise Exception("Missing data")
        if self.wavelength is not None and self.spectrum is not None:
            from ..spectrum import wavelength_to_frequency

            return wavelength_to_frequency(1e9 * self.wavelength, self.spectrum)
        raise Exception("Missing data")

    def to_normalized(self):
        """Returns a normalized version of the current instance."""
        normalized_spectrum = self.spectrum / np.max(self.spectrum)

        return IntensitySpectrum(
            spectrum=normalized_spectrum,
            phase=self.phase,
            freq=np.array(self.freq),
            wavelength=np.array(self.wavelength),
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def to_interpolated_wavelength(self, new_wavelengths: np.ndarray):
        """Replace the wavelength axis and interpolate the rest of the data onto it."""
        new_spectrum = interpolate(new_wavelengths, self.wavelength, self.spectrum)
        if self.phase is not None:
            new_phase = interpolate(new_wavelengths, self.wavelength, self.phase)
        else:
            new_phase = None

        return IntensitySpectrum(
            spectrum=new_spectrum,
            wavelength=new_wavelengths,
            freq=constants.speed_of_light / new_wavelengths,
            phase=new_phase,
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def to_corrected_wavelength(self, new_wavelengths: np.ndarray):
        """Correct the wavelength axis by replacing it with new wavelengths."""
        assert len(new_wavelengths) == len(self.wavelength)
        return IntensitySpectrum(
            spectrum=self.spectrum,
            wavelength=new_wavelengths,
            freq=constants.speed_of_light / new_wavelengths,
            phase=self.phase,
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def plot_with_group_delay(
        self,
        ax: Optional[Axes] = None,
        phase_blanking: float = 0.05,
        shift_from_centered=True,
        xlim=None,
    ):
        """Plot the spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            shift_from_centered (bool): if the pulse is centered, the group delay will be near +/- the grid length, tells whether to fix this.
            xlim: pass arguments to set_xlim() to constrain the x-axis

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        start_index = np.argmax(self.spectrum > 0)
        intensity = self.spectrum[start_index::]
        freq = self.freq[start_index::]
        wl = constants.speed_of_light / freq
        if self.phase is not None:
            if shift_from_centered:
                shift = -0.5 / (freq[1] - freq[0])
                phase_shift = np.angle(np.exp(1j * 2 * np.pi * freq * shift))
                phase = np.unwrap(phase_shift + self.phase[start_index::])
            else:
                phase = np.unwrap(self.phase[start_index::])

        else:
            phase = np.zeros(intensity.shape, dtype=float)

        intensity /= np.max(intensity)
        intensity_line = ax.plot(1e9 * wl, intensity, label="Intensity")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        ax_phase = plt.twinx(ax)
        group_delay = (1e15 / (2 * np.pi)) * derivative(phase, 1) / (freq[1] - freq[2])
        assert isinstance(ax_phase, Axes)
        ax_phase.plot([], [])
        phase_line = ax_phase.plot(
            1e9 * wl[intensity > phase_blanking],
            group_delay[intensity > phase_blanking],
            "--",
            label="Group delay",
        )
        ax_phase.set_ylabel("Group delay (fs)")
        if xlim is not None:
            ax.set_xlim(xlim)
            ax_phase.set_xlim(xlim)
        lines = lines = intensity_line + phase_line
        ax.legend(lines, [str(line.get_label()) for line in lines])
        return fig
