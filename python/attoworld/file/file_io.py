import scipy.io as sio
import numpy as np

def load_waves_from_matfile(phase: float, filename: str):
    """Load the contents of an attolab scanner file in .mat format

    Args:
        phase (float): phase to use when interpreting the lock-in data
        filename (str): path to the mat file
    Returns:
        time_delay: array of time delay values
        signal: signals corresponding to the time delays
    """
    # Contributed by Nick Karpowicz
    datablob = sio.loadmat(filename)
    stage_position = datablob['xdata'][0,:]
    time_delay = -2e-3 * stage_position/2.9979e8
    lia_x = datablob['x0']
    lia_y = datablob['y0']
    signal = np.fliplr(lia_x*np.cos(phase) - lia_y*np.sin(phase))
    return time_delay, signal
