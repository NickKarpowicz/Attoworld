"""Classes for organizing data from a Frequency Resolved Optical Gating measurement."""

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import constants

from ..plot import label_letter
from .decorators import yaml_io
from .interop import ComplexSpectrum, Waveform
from .spectrogram import Spectrogram


@yaml_io
@dataclass(slots=True)
class FrogData:
    """Stores data from a FROG measurement.

    Attributes:
        spectrum (ComplexSpectrum): the reconstructed complex spectrum
        pulse (Waveform): time-domain reconstructed field
        measured_spectrogram (Spectrogram): measured (binned) data
        reconstructed_spectrogram (Spectrogram): spectrogram resulting from reconstructed field
        raw_reconstruction (np.ndarray): the raw reconstructed time-domain pulse of the FROG
        f0: the central frequency of the spectrum
        dt: the time step

    """

    spectrum: ComplexSpectrum
    pulse: Waveform
    measured_spectrogram: Spectrogram
    reconstructed_spectrogram: Spectrogram
    raw_reconstruction: np.ndarray
    f0: float
    dt: float

    def lock(self):
        """Make the data immutable."""
        self.raw_reconstruction.setflags(write=False)
        self.spectrum.lock()
        self.pulse.lock()
        self.measured_spectrogram.lock()
        self.reconstructed_spectrogram.lock()

    def save(self, base_filename):
        """Save in the Trebino FROG format.

        Args:
            base_filename: base of the file path; 4 files will be made from it: .A.dat, .Arecon.dat, .Ek.dat, and .Speck.dat

        """
        self.measured_spectrogram.save(base_filename + ".A.dat")
        self.reconstructed_spectrogram.save(base_filename + ".Arecon.dat")

        t = 1e15 * self.dt * np.array(range(len(self.raw_reconstruction)))
        t -= np.mean(t)
        f = np.fft.fftshift(np.fft.fftfreq(len(t), d=self.dt)) + self.f0
        lam = constants.speed_of_light / f
        raw_spec = np.fft.fftshift(np.fft.fft(self.raw_reconstruction))

        with open(base_filename + ".Ek.dat", "w") as time_file:
            for _i in range(len(self.raw_reconstruction)):
                time_file.write(
                    f"{t[_i]:.15g}\t{np.abs(self.raw_reconstruction[_i]) ** 2:.15g}\t{np.angle(self.raw_reconstruction[_i]):.15g}\t{np.real(self.raw_reconstruction[_i]):.15g}\t{np.imag(self.raw_reconstruction[_i]):.15g}\n"
                )

        with open(base_filename + ".Speck.dat", "w") as spec_file:
            for _i in range(len(self.raw_reconstruction)):
                spec_file.write(
                    f"{lam[_i]:.15g}\t{np.abs(raw_spec[_i]) ** 2:.15g}\t{np.angle(raw_spec[_i]):.15g}\t{np.real(raw_spec[_i]):.15g}\t{np.imag(raw_spec[_i]):.15g}\n"
                )

    def plot_measured_spectrogram(self, ax: Optional[Axes] = None, log: bool = False):
        """Plot the measured spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            log (bool): plot on log scale

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if log:
            self.measured_spectrogram.plot_log(ax)
        else:
            self.measured_spectrogram.plot(ax)
        ax.set_title("Measured")
        return fig

    def plot_reconstructed_spectrogram(
        self, ax: Optional[Axes] = None, log: bool = False
    ):
        """Plot the reconstructed spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            log (bool): plot on log scale

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if log:
            self.reconstructed_spectrogram.plot_log(ax)
        else:
            self.reconstructed_spectrogram.plot(ax)
        ax.set_title(
            f"Retrieved (G': {self.get_error():0.2e}; G: {self.get_G_error():0.2e})"
        )
        return fig

    def plot_pulse(
        self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None
    ):
        """Plot the reconstructed pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis

        """
        return self.pulse.to_complex_envelope().plot(ax, phase_blanking, xlim)

    def plot_spectrum(
        self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None
    ):
        """Plot the reconstructed spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis

        """
        return self.spectrum.to_intensity_spectrum().plot_with_group_delay(
            ax, phase_blanking=phase_blanking, shift_from_centered=True, xlim=xlim
        )

    def plot_all(
        self,
        phase_blanking=0.05,
        time_xlims=None,
        wavelength_autoscale=1e-3,
        wavelength_xlims=None,
        figsize=None,
        log: bool = False,
    ):
        """Produce a 4-panel plot of the FROG results, combining calls to plot_measured_spectrogram(),
        plot_reconstructed_spectrogram(), plot_pulse() and plot_spectrum() as subplots, with letter labels.

        Args:
            phase_blanking: relative intensity at which to show phase information
            time_xlims: x-axis limits to pass to the plot of the pulse
            wavelength_autoscale: intensity relative to the peak to include within the spectrum (overrides wavelength_xlims unless set to None)
            wavelength_xlims: x-axis limits to pass to the plot of the spectrum
            figsize: custom figure size
            log (bool): plot on log scale

        """
        if figsize is None:
            default_figsize = plt.rcParams["figure.figsize"]
            figsize = (default_figsize[0] * 2, default_figsize[1] * 2)
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        self.plot_measured_spectrogram(ax[0, 0], log=log)
        label_letter("a", ax[0, 0])
        self.plot_reconstructed_spectrogram(ax[1, 0], log=log)
        label_letter("b", ax[1, 0])
        self.plot_pulse(ax[0, 1], xlim=time_xlims, phase_blanking=phase_blanking)
        label_letter("c", ax[0, 1])
        if wavelength_autoscale is not None:
            if not isinstance(wavelength_xlims, tuple):
                spec = self.spectrum.to_intensity_spectrum()
                wl_nm = spec.wavelength_nm()
                indices = np.where(
                    spec.spectrum / np.max(spec.spectrum) > wavelength_autoscale
                )[0]
                wavelength_xlims = (wl_nm[indices[-1]], wl_nm[indices[0]])
        self.plot_spectrum(
            ax[1, 1], xlim=wavelength_xlims, phase_blanking=phase_blanking
        )
        label_letter("d", ax[1, 1])
        return fig

    def get_error(self) -> float:
        """Get the G' error of the reconstruction."""
        norm_measured = np.linalg.norm(self.measured_spectrogram.data)
        norm_retrieved = np.linalg.norm(self.reconstructed_spectrogram.data)
        return np.sqrt(
            np.sum(
                (
                    self.measured_spectrogram.data[:] / norm_measured
                    - self.reconstructed_spectrogram.data[:] / norm_retrieved
                )
                ** 2
            )
            / np.sum((self.measured_spectrogram.data[:] / norm_measured) ** 2)
        )

    def get_G_error(self) -> float:
        """Get the G (note: no apostrophe) error. This one doesn't mean much, but is useful
        for comparing reconstructions of the same spectrogram between different programs.
        """
        return np.sqrt(
            (1.0 / float(len(self.measured_spectrogram.data[:]) ** 2))
            * np.sum(
                (
                    self.measured_spectrogram.data[:]
                    / np.max(self.measured_spectrogram.data[:])
                    - self.reconstructed_spectrogram.data[:]
                    / np.max(self.reconstructed_spectrogram.data[:])
                )
                ** 2
            )
        )

    def get_fwhm(self) -> float:
        """Get the full-width-at-half-max value of the reconstructed pulse."""
        return self.pulse.get_envelope_fwhm()
