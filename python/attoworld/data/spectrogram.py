"""Contain data describing a spectrogram."""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import constants

from ..numeric import (
    block_binning_1d,
    block_binning_2d,
    interpolate,
)
from .decorators import yaml_io


@yaml_io
@dataclass(slots=True)
class FrogBinSettings:
    """Stores the settings for binning a FROG measurement."""

    size: int
    dt: float
    f0: float
    dc_offset: float
    time_binning: int
    freq_binning: int
    median_binning: bool
    spatial_chirp_correction: bool


@yaml_io
@dataclass(slots=True)
class Spectrogram:
    """Contains the data describing a spectrogram.

    Attributes:
        data (np.ndarray): 2d spectrogram
        time (np.ndarray): time vector
        freq (np.ndarray): frequency vector

    """

    data: np.ndarray
    time: np.ndarray
    freq: np.ndarray

    def lock(self):
        """Make the data immutable."""
        self.data.setflags(write=False)
        self.time.setflags(write=False)
        self.freq.setflags(write=False)

    def save(self, filename):
        """Save in the .A.dat file format used by FROG .etc
        Args:
            filename: the file to be saved.

        The file is structured like this:
        [number of wavelengths] [number of times]
        [minimum value of the trace] [maximum value of the trace]
        [array of wavelengths]
        [array of times]
        [data array as single column]
        """
        with open(filename, "w") as file:
            file.write(f"{len(self.freq)}\t{len(self.time)}\n")
            file.write(f"{np.min(self.data[:])}\t{np.max(self.data[:])}\n")
            for freq in self.freq:
                wavelength_nm = 1e9 * constants.speed_of_light / freq
                file.write(f"{wavelength_nm:15.15g}\n")
            for time in self.time:
                time_fs = 1e15 * time
                file.write(f"{time_fs:15.15g}\n")
            for x in self.data:
                for y in x:
                    file.write(f"{y:15.15g}\n")

    def to_block_binned(self, freq_bin: int, time_bin: int, method: str = "mean"):
        """Apply block-binning to the spectrogram.

        Args:
            freq_bin (int): block size for averaging in the frequency direction
            time_bin (int): block size for averaging in the time-direction
            method (str): can be ```mean``` or ```median```

        """
        return Spectrogram(
            data=block_binning_2d(self.data, time_bin, freq_bin, method),
            freq=block_binning_1d(self.freq, freq_bin, "mean"),
            time=block_binning_1d(self.time, time_bin, "mean"),
        )

    def to_per_frequency_dc_removed(self, extra_offset: float = 0.0):
        """Perform DC offset removal on a measured spectrogram, on a per-frequency basis.

        Args:
            extra_offset (float): subtract a value from the entire array (negative values are always set to zero)

        Returns:
            Spectrogram: the spectrogram with offset removed.

        """
        new_data = np.array(self.data)
        new_data -= extra_offset
        new_data[new_data < 0.0] = 0.0
        for _i in range(new_data.shape[0]):
            new_data[_i, :] -= np.min(new_data[_i, :])

        return Spectrogram(data=new_data, time=self.time, freq=self.freq)

    def to_symmetrized(self):
        """Average the trace with a time-reversed copy. This might be useful for getting a reconstruction of difficult data, but keep in mind that the resulting measured trace will no longer represent the real measurement and should not be published as such."""
        return Spectrogram(
            data=0.5 * (self.data + np.fliplr(self.data)),
            time=self.time,
            freq=self.freq,
        )

    def to_removed_spatial_chirp(self):
        """Remove the effects of spatial chirp on an SHG-FROG trace by centering all single-frequency autocorrelations to the same time-zero."""
        new_data = np.array(self.data)
        for i in range(len(self.freq)):
            total = np.sum(self.data[i, :])
            if total > 0.0:
                t0 = np.sum(self.time * self.data[i, :]) / total
                new_data[i, :] = interpolate(self.time + t0, self.time, self.data[i, :])

        return Spectrogram(data=new_data, time=self.time, freq=self.freq)

    def to_combined_and_binned(
        self,
        other,
        stitching_band: Tuple[float, float],
        dim: int = 64,
        dt: float = 5e-15,
        t0: Optional[Tuple[float, float]] = None,
        f0: float = 750e12,
    ):
        """Bin two different spectrograms, e.g. from different spectrometers, onto the time time/frequency grid.

        Args:
            other: the other spectrogram
            stitching_band (Tuple[float, float]): the lower and upper frequency of the band where the two spectrometers should have equivalent response (hopefully there is one)
            dim (int): size of each size of the resulting square data
            dt (float): time step of the data
            t0: (Optional[Tuple[float, float]): time-zero of the data (this, and other). If not specified, will be calculated by the first moment of the time-distribution of the signal
            f0: (float): central frequency of the binned array

        """
        t0_self = None
        t0_other = None

        if t0 is not None:
            t0_self = t0[0]
            t0_other = t0[1]

        binned_self = self.to_binned(dim, dt, t0_self, f0)
        binned_other = other.to_binned(dim, dt, t0_other, f0)
        freq = binned_self.freq

        # add more logic here to combine the spectrograms
        stitching_band_integral_self = np.sum(
            binned_self.data[
                ((freq > stitching_band[0]) & (freq < stitching_band[1])), :
            ][:]
        )
        stitching_band_integral_other = np.sum(
            binned_other.data[
                ((freq > stitching_band[0]) & (freq < stitching_band[1])), :
            ][:]
        )
        weights_self = np.zeros(binned_self.freq.shape, dtype=float)
        weights_other = np.zeros(binned_other.freq.shape, dtype=float)
        other_multiplier = stitching_band_integral_self / stitching_band_integral_other
        for i in range(len(freq)):
            sum_self = np.sum(binned_self.data[i, :])
            sum_other = other_multiplier * np.sum(binned_other.data[i, :])
            total = sum_self + sum_other
            if total > 0.0:
                weight_self = sum_self / total
                weight_other = other_multiplier * sum_other / total
                weights_self[i] = weight_self
                weights_other[i] = weight_other

        return Spectrogram(
            data=weights_self[:, np.newaxis] * binned_self.data
            + weights_other[:, np.newaxis] * binned_other.data,
            time=binned_self.time,
            freq=binned_self.freq,
        )

    def to_binned(
        self,
        dim: int = 64,
        dt: float = 5e-15,
        t0: Optional[float] = None,
        f0: float = 750e12,
    ):
        """Bin a spectrogram to a FFT-appropriate shape.

        Args:
            dim (int): size of each size of the resulting square data
            dt (float): time step of the data
            t0: (Optional[float]): time-zero of the data. If not specified, will be calculated by the first moment of the time-distribution of the signal
            f0: (float): central frequency of the binned array

        Returns:
            Spectrogram: the binned spectrogram

        """
        _t = np.array(range(dim)) * dt
        _t -= np.mean(_t)
        _f = np.fft.fftshift(np.fft.fftfreq(dim, d=dt) + f0)
        binned_data = np.zeros((dim, self.time.shape[0]), dtype=float)
        for _i in range(self.time.shape[0]):
            binned_data[:, _i] = interpolate(
                _f, self.freq, np.array(self.data[:, _i]), neighbors=2
            )
        binned_data /= np.max(binned_data[:])
        if t0 is None:
            ac = np.sum(binned_data, axis=0)
            t0 = np.sum(ac * self.time) / np.sum(ac)
        binned_data_square = np.zeros((dim, dim), dtype=float)
        for _i in range(dim):
            binned_data_square[_i, :] = interpolate(
                _t, self.time - t0, np.array(binned_data[_i, :]), neighbors=2
            )
        return Spectrogram(data=binned_data_square, time=_t, freq=_f)

    def to_bin_pipeline_result(self, settings: FrogBinSettings):
        """Apply a FrogBinSettings dataclass to the spectrogram.

        Args:
            settings (FrogBinSettings): the dataclass to apply

        """
        if settings.median_binning:
            method = "media"
        else:
            method = "mean"

        def maybe_correct_chirp(instance, apply: bool):
            return instance.to_removed_spatial_chirp() if apply else instance

        return (
            maybe_correct_chirp(self, settings.spatial_chirp_correction)
            .to_block_binned(settings.freq_binning, settings.time_binning, method)
            .to_binned(dim=settings.size, dt=settings.dt, f0=settings.f0)
            .to_per_frequency_dc_removed(extra_offset=settings.dc_offset)
        )

    def plot(self, ax: Optional[Axes] = None):
        """Plot the spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        a = ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            self.data / np.max(self.data[:]),
            rasterized=True,
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Frequency (THz)")
        plt.colorbar(a)
        return fig

    def plot_log(self, ax: Optional[Axes] = None):
        """Plot the spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        logdata = np.array(self.data)
        logdata[self.data > 0.0] = np.log(self.data[self.data > 0.0])
        logdata[self.data < 0.0] = 0.0
        a = ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            logdata,
            rasterized=True,
            vmin=-11,
            vmax=0,
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Frequency (THz)")
        ax.grid(True, lw=1)
        plt.colorbar(a)
        return fig
