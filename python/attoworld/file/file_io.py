import scipy.io as sio
import numpy as np
import pandas
from typing import Optional

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
