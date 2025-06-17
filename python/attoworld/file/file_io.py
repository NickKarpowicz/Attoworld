import scipy.io as sio
from scipy import constants
import numpy as np
import pandas
from ..numeric import interpolate
from pathlib import Path
from typing import Optional
from .dataclasses import Waveform, IntensitySpectrum, Spectrogram, FrogData, ComplexSpectrum

def load_spectrum_from_text(filename: str, wavelength_multiplier: float = 1e-9, wavelength_field:str='wavelength (nm)', spectrum_field:str='intensity (a.u.)', sep:str='\t'):
    """
    Load a spectrum contained in a text file.

    Args:
        filename (str): path to the file
        wavelength_multiplier (float): multiplier to convert wavelength to m
        wavelength_field (str): name of the field in the data corresponding to wavelength
        spectrum_field (str): name of the field in the data corresponding to spectral intensity
        sep (str): column separator
    Returns:
        IntensitySpectrum: the intensity spectrum"""
    data = pandas.read_csv(filename, sep=sep)
    wavelength = wavelength_multiplier * np.array(data[wavelength_field])
    freq = constants.speed_of_light / wavelength
    spectrum = np.array(data[spectrum_field])
    return IntensitySpectrum(spectrum=spectrum, wavelength=wavelength, freq=freq)

def load_waveform_from_text(filename: str, time_multiplier: float = 1e-15, time_field:str='delay (fs)', wave_field:str='field (a.u.)', sep='\t') -> Waveform:
    """Loads a waveform from a text file

    Args:
        filename (str): path to the file
        time_multiplier (float): multiplier needed to convert the time unit in the file to seconds
        time_field (str): name (header) of the column containing the times
        wave_field (str): name (header) of the column containing the waveform
        sep (str): separator used in the file format (tab is default)

    Returns:
        Waveform: the waveform
    """

    data = pandas.read_csv(filename, sep=sep)
    time = time_multiplier * data[time_field].to_numpy()
    wave = data[wave_field].to_numpy()
    dt = time[1]-time[0]
    diff_time = np.diff(time)
    uniform = bool(np.all(np.isclose(diff_time, diff_time[0])))
    return Waveform(wave = wave, time = time, dt=dt, is_uniformly_spaced = uniform)


def load_waves_from_matfile(filename: str, phase: Optional[float] = None):
    """Load the contents of an attolab scanner file in .mat format

    Args:
        phase (float): phase to use when interpreting the lock-in data
        filename (str): path to the mat file
    Returns:
        time_delay: array of time delay values
        signal: signals corresponding to the time delays
    """

    datablob = sio.loadmat(filename)
    stage_position = datablob['xdata'][0,:]
    time_delay = -2e-3 * stage_position/2.9979e8
    lia_x = datablob['x0']
    lia_y = datablob['y0']
    if phase is None:
        optimized_phase = np.atan2(np.sum(lia_y[:]**2), np.sum(lia_x[:]**2))
        signal = np.fliplr(lia_x*np.cos(optimized_phase) + lia_y*np.sin(optimized_phase))
        return time_delay, signal
    else:
        signal = np.fliplr(lia_x*np.cos(phase) - lia_y*np.sin(phase))
        return time_delay, signal


def read_Trebino_FROG_matrix(filename: Path | str) -> Spectrogram:
    """
    Read a spectrogram file made by the Trebino FROG code

    Args:
        filename (Path | str): the name (path) of the file
    """
    with open(filename, "r") as f:
        l = str(f.readline())
        l = l.split()
        n1 = int(l[0])
        n2 = int(l[1])
        l = str(f.readline())
        l = l.split()
    measured_data = pandas.read_csv(filename, sep='\t', header = None, skiprows=2)
    measure = []
    raw_freq = 1e9*constants.speed_of_light/np.array(measured_data[0][0:n2]).squeeze()
    df = np.mean(np.diff(raw_freq))
    freq = raw_freq[0] + df * np.array(range(raw_freq.shape[0]))
    time = 1e-15 * np.array(measured_data[0][n2:(n2+n1)]).squeeze()
    for i in range(n1):
        measure.append(measured_data[0][(i+2)*n2:(i+3)*n2])
    data = np.array(measure)
    return Spectrogram(data = data, time = time, freq = freq)

def read_Trebino_FROG_speck(filename: Path | str) -> ComplexSpectrum:
    """
    Read a .Speck file made by the Trebino FROG code

    Args:
        filename (Path | str): the name (path) of the file
    """
    data = np.array(pandas.read_csv(filename, sep='\t', header = None), dtype=float)
    raw_freq = 1e9*constants.speed_of_light/data[:,0]
    df = np.mean(np.diff(raw_freq))
    freq = np.linspace(0.0, raw_freq[-1], int(np.ceil(raw_freq[-1]/df)))
    spectrum = interpolate(freq, raw_freq, data[:,3]) + 1j * interpolate(freq, raw_freq, data[:,4])
    return ComplexSpectrum(spectrum=spectrum, freq=freq)

def read_Trebino_FROG_data(filename: str) -> FrogData:
    """
    Read a set of data produced by the Trebino FROG reconstruction code.

    Args:
        filename: Base filename of the .bin file; e.g. if the data are mydata.bin.Speck.dat etc., this will be "mydata.bin" """
    spectrum = read_Trebino_FROG_speck(filename+'.Speck.dat')
    pulse = spectrum.to_centered_waveform()
    measured_spectrogram = read_Trebino_FROG_matrix(filename+'.A.dat')
    reconstructed_spectrogram = read_Trebino_FROG_matrix(filename+'.Arecon.dat')
    return FrogData(spectrum = spectrum,
        pulse = pulse,
        measured_spectrogram = measured_spectrogram,
        reconstructed_spectrogram = reconstructed_spectrogram)
