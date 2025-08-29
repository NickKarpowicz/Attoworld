"""Functions for accessing and saving data."""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import constants

from .. import spectrum
from ..numeric import interpolate
from .frog_data import FrogData, Spectrogram
from .interop import ComplexSpectrum, IntensitySpectrum, Waveform


def read_dwc(file_or_path, is_buffer: bool = False):
    """Reads files in the .dwc format produced by many FROG scanners.

    Args:
        file_or_path: path to the .dwc file
        is_buffer: set to true if you are giving a file in a bytes buffer instead of a path

    Returns:
        Spectrogram: the loaded data

    """
    if is_buffer:
        file_or_path = io.BytesIO(file_or_path)
        lines = file_or_path.readlines()
        delay_increment = float(lines[2].decode().strip().split("=")[1])
        file_or_path.seek(0)
    else:
        with open(file_or_path, "r") as file:
            lines = file.readlines()
            delay_increment = float(lines[2].strip().split("=")[1])

    wavelength_vector = pd.read_csv(
        file_or_path, skiprows=5, nrows=1, delimiter="\t", header=None
    ).values[0]

    if is_buffer:
        file_or_path.seek(0)

    data_array = pd.read_csv(
        file_or_path, skiprows=8, delimiter="\t", header=None, dtype=float
    ).values

    freqs, first_spec = spectrum.wavelength_to_frequency(
        wavelength_vector, data_array[:, 0]
    )
    if freqs is not None:
        data_array_freq = np.zeros((len(freqs), data_array.shape[1]))
        data_array_freq[:, 0] = first_spec
        for _i in range(data_array.shape[1]):
            _f, _dat = spectrum.wavelength_to_frequency(
                wavelength_vector, data_array[:, _i], freqs
            )
            data_array_freq[:, _i] = _dat
        dt = 1e-15 * delay_increment
        delays = dt * np.array(range(data_array.shape[1]))
        delays -= np.mean(delays)
        return Spectrogram(data=data_array_freq, time=delays, freq=freqs)
    raise Exception("Interpolation failure reading dwc file")


def load_mean_spectrum_from_scarab(filename: str):
    """Load data produced by Scarab (Nick's C++ interface for Ocean Optics spectrometers)."""
    data = np.loadtxt(filename)
    return IntensitySpectrum(
        spectrum=np.mean(data[:, 1::], axis=1),
        wavelength=1e-9 * data[:, 0],
        freq=1e9 * constants.speed_of_light / data[:, 0],
        is_frequency_scaled=False,
        phase=None,
    )


def load_spectrum_from_text(
    filename: str,
    wavelength_multiplier: float = 1e-9,
    wavelength_field: str = "wavelength (nm)",
    spectrum_field: str = "intensity (a.u.)",
    sep: str = "\t",
):
    """Load a spectrum contained in a text file.

    Args:
        filename (str): path to the file
        wavelength_multiplier (float): multiplier to convert wavelength to m
        wavelength_field (str): name of the field in the data corresponding to wavelength
        spectrum_field (str): name of the field in the data corresponding to spectral intensity
        sep (str): column separator
    Returns:
        IntensitySpectrum: the intensity spectrum

    """
    data = pd.read_csv(filename, sep=sep)
    wavelength = wavelength_multiplier * np.array(data[wavelength_field])
    freq = constants.speed_of_light / wavelength
    spectrum = np.array(data[spectrum_field])
    return IntensitySpectrum(
        spectrum=spectrum,
        wavelength=wavelength,
        freq=freq,
        phase=np.zeros(spectrum.shape, dtype=float),
    )


def load_waveform_from_text(
    filename: str,
    time_multiplier: float = 1e-15,
    time_field: str = "delay (fs)",
    wave_field: str = "field (a.u.)",
    sep="\t",
) -> Waveform:
    """Loads a waveform from a text file.

    Args:
        filename (str): path to the file
        time_multiplier (float): multiplier needed to convert the time unit in the file to seconds
        time_field (str): name (header) of the column containing the times
        wave_field (str): name (header) of the column containing the waveform
        sep (str): separator used in the file format (tab is default)

    Returns:
        Waveform: the waveform

    """
    data = pd.read_csv(filename, sep=sep)
    time = time_multiplier * data[time_field].to_numpy()
    wave = data[wave_field].to_numpy()
    dt = time[1] - time[0]
    diff_time = np.diff(time)
    uniform = bool(np.all(np.isclose(diff_time, diff_time[0])))
    return Waveform(wave=wave, time=time, dt=dt, is_uniformly_spaced=uniform)


def load_waves_from_matfile(filename: str, phase: Optional[float] = None):
    """Load the contents of an attolab scanner file in .mat format.

    Args:
        phase (float): phase to use when interpreting the lock-in data
        filename (str): path to the mat file
    Returns:
        time_delay: array of time delay values
        signal: signals corresponding to the time delays

    """
    datablob = sio.loadmat(filename)
    stage_position = datablob["xdata"][0, :]
    time_delay = -2e-3 * stage_position / 2.9979e8
    lia_x = datablob["x0"]
    lia_y = datablob["y0"]
    if phase is None:
        optimized_phase = np.atan2(np.sum(lia_y[:] ** 2), np.sum(lia_x[:] ** 2))
        signal = np.fliplr(
            lia_x * np.cos(optimized_phase) + lia_y * np.sin(optimized_phase)
        )
        return time_delay, signal
    signal = np.fliplr(lia_x * np.cos(phase) - lia_y * np.sin(phase))
    return time_delay, signal


def read_Trebino_FROG_matrix(filename: Path | str) -> Spectrogram:
    """Read a spectrogram file made by the Trebino FROG code.

    Args:
        filename (Path | str): the name (path) of the file

    """
    with open(filename, "r") as f:
        line = str(f.readline())
        line = line.split()
        n1 = int(line[0])
        n2 = int(line[1])
        line = str(f.readline())
        line = line.split()
    measured_data = pd.read_csv(filename, sep="\t", header=None, skiprows=2)
    measure = []
    raw_freq = (
        1e9 * constants.speed_of_light / np.array(measured_data[0][0:n2]).squeeze()
    )
    df = np.mean(np.diff(raw_freq))
    freq = raw_freq[0] + df * np.array(range(raw_freq.shape[0]))
    time = 1e-15 * np.array(measured_data[0][n2 : (n2 + n1)]).squeeze()
    for i in range(n1):
        measure.append(measured_data[0][(i + 2) * n2 : (i + 3) * n2])
    data = np.array(measure)
    return Spectrogram(data=data, time=time, freq=freq)


def read_Trebino_FROG_speck(filename: Path | str) -> ComplexSpectrum:
    """Read a .Speck file made by the Trebino FROG code.

    Args:
        filename (Path | str): the name (path) of the file

    """
    data = np.array(pd.read_csv(filename, sep="\t", header=None), dtype=float)
    raw_freq = 1e9 * constants.speed_of_light / data[:, 0]
    df = np.mean(np.diff(raw_freq))
    freq = np.linspace(0.0, raw_freq[-1], int(np.ceil(raw_freq[-1] / df)))
    spectrum = interpolate(freq, raw_freq, data[:, 3]) + 1j * interpolate(
        freq, raw_freq, data[:, 4]
    )
    return ComplexSpectrum(spectrum=spectrum, freq=freq)


def read_Trebino_FROG_data(filename: str) -> FrogData:
    """Read a set of data produced by the Trebino FROG reconstruction code.

    Args:
        filename: Base filename of the .bin file; e.g. if the data are mydata.bin.Speck.dat etc., this will be "mydata.bin"

    """
    spectrum = read_Trebino_FROG_speck(filename + ".Speck.dat")
    pulse = spectrum.to_centered_waveform()
    measured_spectrogram = read_Trebino_FROG_matrix(filename + ".A.dat")
    reconstructed_spectrogram = read_Trebino_FROG_matrix(filename + ".Arecon.dat")
    raw_speck = np.array(
        pd.read_csv(filename + ".Speck.dat", sep="\t", header=None), dtype=float
    )
    raw_ek = np.array(
        pd.read_csv(filename + ".Ek.dat", sep="\t", header=None), dtype=float
    )
    f0 = 1e9 * np.mean(constants.speed_of_light / raw_speck[:, 0])
    dt = 1e-15 * (raw_ek[1, 0] - raw_ek[0, 0])
    raw_reconstruction = raw_ek[:, 3] + 1.0j * raw_ek[:, 4]
    return FrogData(
        spectrum=spectrum,
        pulse=pulse,
        measured_spectrogram=measured_spectrogram,
        reconstructed_spectrogram=reconstructed_spectrogram,
        raw_reconstruction=raw_reconstruction,
        dt=dt,
        f0=float(f0),
    )
