import scipy.io as sio
from scipy import constants
import numpy as np
import pandas
from typing import Optional
from .dataclasses import Waveform, IntensitySpectrum

def load_spectrum_from_text(filename: str, wavelength_multiplier: float = 1e-9, wavelength_field:str='wavelength (nm)', spectrum_field:str='intensity (a.u.)', sep:str='\t'):
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
