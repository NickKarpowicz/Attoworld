"""Class for handling a complex pulse envelope."""

import copy
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..numeric import derivative, fwhm
from .decorators import yaml_io


@yaml_io
@dataclass(slots=True)
class ComplexEnvelope:
    """Data corresponding to a complex envelope of a pulse, e.g. from a FROG measurement.

    Attributes:
        envelope (np.ndarray): the complex envelope
        time: (np.ndarray): the time array
        dt (float): the time step
        carrier_frequency (float): the carrier frequency of the envelope

    """

    envelope: np.ndarray
    time: np.ndarray
    dt: float
    carrier_frequency: float = 0.0

    def lock(self):
        """Make the data immutable."""
        self.envelope.setflags(write=False)
        self.time.setflags(write=False)

    def time_fs(self):
        """Time axis in femtoseconds."""
        return 1e15 * self.time

    def copy(self):
        """Return a copy."""
        return copy.deepcopy(self)

    def get_fwhm(self) -> float:
        """Full-width-at-half-maximum value of the envelope
        Returns:
            float: the fwhm.
        """
        return fwhm(np.abs(self.envelope) ** 2, self.dt)

    def plot(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """Plot the pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        time_ax = self.time_fs() - np.mean(self.time_fs())
        intensity = np.abs(self.envelope) ** 2
        intensity /= np.max(intensity)
        intensity_line = ax.plot(
            time_ax,
            intensity,
            label=f"Intensity, fwhm {1e15 * self.get_fwhm():0.1f} fs",
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Intensity (Arb. unit)")
        inst_freq = (
            (1e-12 / (2 * np.pi))
            * derivative(np.unwrap(np.angle(self.envelope)), 1)
            / self.dt
        )
        ax_phase = plt.twinx(ax)
        assert isinstance(ax_phase, Axes)
        ax_phase.plot([], [])
        phase_line = ax_phase.plot(
            time_ax[intensity > phase_blanking],
            inst_freq[intensity > phase_blanking],
            "--",
            label="Inst. frequency",
        )
        ax_phase.set_ylabel("Inst. frequency (THz)")
        if xlim is not None:
            ax.set_xlim(xlim)
            ax_phase.set_xlim(xlim)
        lines = lines = intensity_line + phase_line
        ax.legend(lines, [str(line.get_label()) for line in lines])
        return fig
