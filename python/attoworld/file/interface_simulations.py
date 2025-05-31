import numpy as np
import scipy.signal
import h5py

e = 1.602176462e-19
hbar = 1.05457159682e-34
m = 9.1093818872e-31
eps0 = 8.854187817e-12
kB = 1.380650324e-23
c = 299792458

figureSize = [30,12]
tickSize = 48
fontSize = 55
lineWidth = 6
legendFontSize = 41
axisLineWidth = 3

def check_equal_length(*arg):

    n = len(arg[0])
    for v in arg:
        if len(v) != n:
            print(v)
            print('Error: vector size mismatch')
            raise Exception('Vector size mismatch')

def fourier_transform(TimeV, FieldV):   # complex!!!

    freq = np.fft.fftfreq(TimeV.size, d=TimeV[1]-TimeV[0])
    fft = np.fft.fft(FieldV)
    return freq, fft

def inverse_fourier_transform(freq, fullSpectrum):  # complex!!!

    timeV = np.fft.fftfreq(len(freq), freq[1]-freq[0])
    if len(timeV)%2 == 0:
        timeV = np.concatenate((timeV[int(len(timeV)/2):], timeV[:int(len(timeV)/2)]))
    else:
        timeV = np.concatenate((timeV[int((len(timeV)+1)/2):], timeV[:int((len(timeV)+1)/2)]))
    fieldV = np.fft.ifft(fullSpectrum)

    return timeV, fieldV

class LunaResult:
    """Loads and handles the Luna simulation result.

    The result must be in the HDF5 format using the saving option in the Luna.Interface.prop_capillary() function [filepath="..."].
    As opposed to most of the analysis routines here, units are in SI!"""
    def __init__(self, filename):
        self.filename = filename
        self.fieldFT = None
        self.omega = None
        self.stats = None
        self.z = None
        self.grid = None
        self.open_Luna_result(filename)

    def open_Luna_result(self, filename):
        """Opens the Luna result file and loads the data"""
        with h5py.File(filename, 'r') as f:
            data = f
            self.fieldFT = np.array(data['Eω'])
            self.grid = data['grid']
            self.omega = np.array(self.grid['ω'])
            self.z = np.array(data['z'])
            self.stats = data['stats']

    def average_modes(self):
        """Averages the propagation modes in the Luna result file"""
        if len(self.fieldFT.shape) == 3:
            self.fieldFT = np.mean(self.fieldFT, axis=1)

    def select_mode(self, mode: int):
        if len(self.fieldFT.shape) < 3:
            print("WARNING: No mode to select")
        elif mode > self.fieldFT.shape[1] or mode < 0:
            print("WARNING: mode ", mode, " is out of range")
        else:
            self.fieldFT = self.fieldFT[:, mode, :]

    def get_time_field(self, position=None):
        """Get the COMPLEX electric field in time from the Luna result file

        Args:
            position (float): position along the fiber in m. If None, the end of the fiber is used.

        Returns:
            timeV (numpy.ndarray): time axis in seconds
            fieldV (numpy.ndarray): electric field in V/m ( COMPLEX!! -> remember to take the real part to get the field or the absolute value to get the envelope
                if you take the imaginary part, I bear no responsibility for that)
        """
        self.average_modes()
        if position is None:
            position = self.z[-1]
        index = np.argmin(np.abs(self.z - position))
        if position > np.max(self.z) or position < np.min(self.z):
            print("WARNING: position ", position, "m is out of range")
        check_equal_length(self.fieldFT[index], self.omega)
        fieldFFT = np.concatenate((self.fieldFT[index, :], np.conjugate(self.fieldFT[index, :][::-1])*0))
        freq = np.concatenate((self.omega, -self.omega[::-1])) / 2 / np.pi
        timeV, fieldV = inverse_fourier_transform(freq, fieldFFT)
        return timeV, fieldV

    def get_wavelength_spectrum(self, position=None):
        """Get the spectrum from the Luna result file (|FFT|^2 * (2 * pi * c / λ^2))

        Args:
            position (float): position along the fiber in m. If None, the end of the fiber is used.

        Returns:
            wvl (numpy.ndarray): wavelength axis in m
            wvlSpectrum (numpy.ndarray): electric field spectrum in V/m ( COMPLEX !! )
        """
        self.average_modes()
        if position is None:
            position = self.z[-1]
        index = np.argmin(np.abs(self.z - position))
        if position > np.max(self.z) or position < np.min(self.z):
            print("WARNING: position ", position, "m is out of range")
        wvl = 2 * np.pi * c / self.omega[::-1]
        wvlSpectrum = np.abs(self.fieldFT[index, ::-1])**2 * (2 * np.pi * c/wvl**2 )
        return wvl, wvlSpectrum


    def get_spectral_phase(self, position=None):
        """Get the spectral phase from the Luna result file

        Args:
            position (float): position along the fiber in m. If None, the end of the fiber is used.

        Returns:
            wvl (numpy.ndarray): wavelength axis in m
            phase (numpy.ndarray): spectral phase in rad
        """
        self.average_modes()
        if position is None:
            position = self.z[-1]
        index = np.argmin(np.abs(self.z - position))
        if position > np.max(self.z) or position < np.min(self.z):
            print("WARNING: position ", position, "m is out of range")
        wvl = 2 * np.pi * c / self.omega[::-1]
        phase = np.angle(self.fieldFT[index, ::-1])
        return wvl, phase



