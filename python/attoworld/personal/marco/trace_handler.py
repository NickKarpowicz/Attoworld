from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.special
from scipy import constants

from ...numeric import find_maximum_location, fwhm


def check_equal_length(*arg):
    n = len(arg[0])
    for v in arg:
        if len(v) != n:
            print(v)
            print("Error: vector size mismatch")
            raise Exception("Vector size mismatch")


def fourier_transform(TimeV, FieldV):  # complex!!!
    freq = np.fft.fftfreq(TimeV.size, d=TimeV[1] - TimeV[0])
    fft = np.fft.fft(FieldV)
    return freq, fft


def inverse_fourier_transform(freq, fullSpectrum):  # complex!!!
    timeV = np.fft.fftfreq(len(freq), freq[1] - freq[0])
    if len(timeV) % 2 == 0:
        timeV = np.concatenate(
            (timeV[int(len(timeV) / 2) :], timeV[: int(len(timeV) / 2)])
        )
    else:
        timeV = np.concatenate(
            (timeV[int((len(timeV) + 1) / 2) :], timeV[: int((len(timeV) + 1) / 2)])
        )
    fieldV = np.fft.ifft(fullSpectrum)

    return timeV, fieldV


def zero_padding(signalTimeV, signalV, fsExtension=50):
    timestep = signalTimeV[1] - signalTimeV[0]
    nPointsExtension = int(fsExtension / timestep)
    fsExtension = nPointsExtension * timestep
    newTimeV = np.linspace(
        signalTimeV[0] - fsExtension, signalTimeV[0] - timestep, nPointsExtension
    )
    newTimeV = np.append(newTimeV, signalTimeV[:])
    newTimeV = np.append(
        newTimeV,
        np.linspace(
            signalTimeV[-1] + timestep, signalTimeV[-1] + fsExtension, nPointsExtension
        )[:],
    )

    newSignalV = np.zeros(nPointsExtension)
    newSignalV = np.append(newSignalV, signalV[:])
    newSignalV = np.append(newSignalV, np.zeros(nPointsExtension)[:])

    trace_length = signalTimeV[-1] - signalTimeV[0]
    trace_center = signalTimeV[0] + trace_length / 2
    newSignalV = newSignalV * np.exp(
        -((newTimeV - trace_center) ** 10) / (trace_length * 9 / 20) ** 10
    )

    return newTimeV, newSignalV


def asymmetric_tukey_f(
    x: float, edge1: float, edge2: float, edge1_width: float, edge2_width: float
):
    if edge1_width < 0 or edge2_width < 0:
        edge1_width = 0
        edge2_width = 0
        print("Warning: negative tukey edge width; rectangular window computed")
    if abs(edge1_width) + abs(edge2_width) > 2 * abs(edge2 - edge1):
        raise Exception("Error: tukey edge width larger than admissible")
    if edge2 <= edge1:
        e1 = edge2
        e2 = edge1
        w1 = edge2_width
        w2 = edge1_width
    elif edge1 < edge2:
        e1 = edge1
        e2 = edge2
        w1 = edge1_width
        w2 = edge2_width
    xmin = e1 - w1 / 2
    xmax = e2 + w2 / 2
    if x < xmin or x > xmax:
        y = 0
    elif xmin <= x and x < xmin + w1:
        y = (1 - np.cos((x - xmin) * np.pi / w1)) / 2
    elif xmin + w1 <= x and x <= xmax - w2:
        y = 1
    elif xmax - w2 < x and x <= xmax:
        y = (1 - np.cos((x - xmax) * np.pi / w2)) / 2
    else:
        print("x, xmax, xmin, w1, w2: ", x, xmax, xmin, w1, w2)
        raise ValueError(
            "in tukey_f, x could not be assigned correctly to the sub intervals, might it (or the other parameters) be NaN?"
        )
    return y


def asymmetric_tukey_window(
    x, edge1: float, edge2: float, edge1_width: float, edge2_width: float
):
    if isinstance(x, np.ndarray):
        y = []
        for xi in x:
            y.append(asymmetric_tukey_f(xi, edge1, edge2, edge1_width, edge2_width))
        y = np.array(y)
    elif isinstance(x, list):
        y = []
        for xi in x:
            y.append(asymmetric_tukey_f(xi, edge1, edge2, edge1_width, edge2_width))
    elif isinstance(x, float):
        y = asymmetric_tukey_f(x, edge1, edge2, edge1_width, edge2_width)
    else:
        raise TypeError(
            "in function tukey window x is neither np array nor list nor float"
        )
    return y


class TraceHandler:
    """Loads and stores the trace (field vs time) saved to file (two columns, tab separated) or given in the form of time and field [or wavelengths, spectrum and phase] arrays.

    ALL THE TIMES ARE IN fs, FREQUENCIES IN PHz, WAVELENGTHS IN nm.

    Most of the class methods modify the object data in place, and don't return anything.

    Attributes:
        fieldTimeV: (WAVEFORM DATA in the time domain) time array (fs)
        fieldV: (WAVEFORM DATA in the time domain) field array
        fieldStdevV: (WAVEFORM DATA in the time domain) field standard deviation (or sigma) array

        frequencyAxis: (WAVEFORM DATA in the frequency domain) frequency axis of the FFT in PHz (should be np.fft compatible, i.e., equally spaced, in the form [0, df, 2*df, ..., fmax, -fmax[-df], ..., -df])
        fftFieldV: (WAVEFORM DATA in the frequency domain) FFT field (complex)
        complexFieldTimeV: (WAVEFORM DATA in the frequency domain) time axis of the complex field (fs)
        complexFieldV: (WAVEFORM DATA in the frequency domain) complex field ( = IFFT{FFT(ω)θ(ω)}(t), where θ(ω) = 1 if ω >= 0, θ(ω) = 0 otherwise )
        wvlAxis: (WAVEFORM DATA in the frequency domain) wavelength axis for the FFT spectrum (nm)
        fftSpectrum: (WAVEFORM DATA in the frequency domain) wavelength-dependent positive FFT spectrum ( = |FFT{field}|^2 * df/dλ, λ > 0 )
        fftphase: (WAVEFORM DATA in the frequency domain) wavelength-dependent spectral phase

        wvlSpectrometer: (SPECTROMETER DATA) wavelength (nm)
        ISpectrometer: (SPECTROMETER DATA) spectral intensity

        fsZeroPadding: (= 150 by default) time span in fs for zero padding (appending zeroes both before and after the trace, in order to get a smoother fft)
        filename: filename to load the trace from
        filename_spectrum: filename to load the spectrum from
        normalization_trace: normalization factor for the trace
        zero_delay: time zero, corresponds to the maximum envelope

    """

    def __init__(
        self,
        filename=None,
        filename_spectrum=None,
        time=None,
        field=None,
        stdev=None,
        wvl=None,
        spectrum=None,
        wvl_FFT_trace=None,
        spectrum_FFT_trace=None,
        phase_FFT_trace=None,
    ):
        """Constructor; loads the trace from file (or from time-field or wavelegth-spectrum-phase arrays) and stores it in the class.

        One of the following must be provided (as a default all args are None):
        * filename: the name of the file containing the trace (two columns, tab separated)
        * time and field: the time vector (fs) and electric field vector (a.u.)
        * wvl_FFT_trace, spectrum_FFT_trace and phase_FFT_trace: the wavelength (nm) spectrum (a.u.) and phase (rad) vectors

        Args:
            filename (str): the name of the file containing the trace (two columns - time and field, tab separated)
            filename_spectrum (str): the name of the file containing the spectrum (two columns - wavelength and spectral intensity, tab separated)
                this does not correspond to the fourier transform of the trace, but to some spectrometer data that you would like to compare to the loaded trace
            time (array): the time vector (fs)
            field (array): the electric field vector (a.u.)
            stdev (array): the standard deviation (or sigma) of the electric field vector (a.u.)
            wvl (array): the wavelength vector (nm) (spectrometer)
            spectrum (array): the spectral intensity vector (a.u.) (spectrometer)
            wvl_FFT_trace (array): the wavelength vector (nm) of the FFT trace (to reconstruct a trace from spectral data)
            spectrum_FFT_trace (array): the spectral intensity vector (a.u.) of the FFT trace (to reconstruct a trace from spectral data)
            phase_FFT_trace (array): the spectral phase vector (rad) of the FFT trace (to reconstruct a trace from spectral data)

        """
        self.fsZeroPadding = 150
        self.filename = filename
        self.filename_spectrum = filename_spectrum

        self.fieldTimeV = None
        self.fieldV = None
        self.fieldStdevV = None

        self.frequencyAxis = None
        self.fftFieldV = None
        self.complexFieldTimeV = None
        self.complexFieldV = None
        self.wvlAxis = None
        self.fftSpectrum = None
        self.fftphase = None

        self.wvlSpectrometer = None
        self.ISpectrometer = None

        self.normalization_trace = None
        self.zero_delay = None

        if filename is not None:
            self.load_trace()
        elif time is not None and field is not None:
            self.load_trace_from_arrays(time, field, stdev)
        elif (
            wvl_FFT_trace is not None
            and spectrum_FFT_trace is not None
            and phase_FFT_trace is not None
        ):
            self.load_trace_from_spectral_data(
                wvl_FFT_trace, spectrum_FFT_trace, phase_FFT_trace
            )
        else:
            print(
                "\n\nWARNING: in function TraceHandler.__init__() no data was loaded\n\n"
            )

        if self.filename_spectrum is not None:
            self.load_spectrum()
        elif wvl is not None and spectrum is not None:
            self.load_spectrum_from_arrays(wvl, spectrum)

    def load_spectrum_from_arrays(self, wvl, spectrum):
        """Loads the spectrum from wavelength and spectrum arrays and stores it in the class.

        The spectrum does not correspond to the fourier transform of the trace, but to some spectrometer data that you would like to compare with the loaded trace

        Args:
            wvl: wavelength array (nm)
            spectrum: spectral intensity array

        """
        """wavelength is in nm"""
        check_equal_length(wvl, spectrum)
        self.wvlSpectrometer = np.array(wvl)
        self.ISpectrometer = np.array(spectrum)

    def load_trace_from_arrays(self, fieldTimeV, fieldV, fieldStdevV=None):
        """Loads the trace from time and field arrays and stores it in the class.

        Args:
            fieldTimeV: time array (fs)
            fieldV: field array
            fieldStdevV: standard deviation - or sigma - to be associated to the trace (optional, default = None)

        """
        if fieldTimeV is None or fieldV is None:
            raise ValueError(
                "\nin function TraceHandler.load_from_arrays() too many arguments were None\n"
                "it might be that the constructor was erroneously used with no or too few arguments\n"
            )
        if np.max(np.abs(fieldTimeV)) < 1.0:
            print(
                "\n\nWARNING: do you remember that the time axis of TraceHandler is in fs?\n\n"
            )
        self.fieldTimeV = np.array(fieldTimeV)
        self.fieldV = np.array(fieldV)
        if fieldStdevV is not None:
            self.fieldStdevV = np.array(fieldStdevV)
        self.normalization_trace = np.max(np.abs(self.fieldV))
        self.update_fft()

    def load_trace(self, fname=None):
        """Loads from file."""
        if fname is not None:
            self.filename = fname
        data = pd.read_csv(self.filename, sep="\t")
        self.fieldTimeV = data["delay (fs)"].to_numpy()
        self.fieldV = data["field (a.u.)"].to_numpy()
        self.fieldStdevV = data["stdev field"].to_numpy()
        self.normalization_trace = np.max(np.abs(self.fieldV))
        self.update_fft()

    def load_trace_from_spectral_data(
        self, wvl_FFT_trace, spectrum_FFT_trace, phase_FFT_trace
    ):
        """Loads the trace from wavelength, spectrum, and spectral phase arrays and stores it in the class.

        This function has a different job than load_spectrum_from_arrays(): in fact, it retrieves the trace from the given spectral data via fourier transform.

        Args:
            wvl_FFT_trace: wavelength array in nm
            spectrum_FFT_trace: (wavelength-)spectral intensity array. This is assumed to be |FFT{field}|^2 * df/dλ
            phase_FFT_trace: spectral phase in rad

        """
        check_equal_length(wvl_FFT_trace, spectrum_FFT_trace, phase_FFT_trace)

        # check that the wvl_FFT_trace array is monotonically increasing
        if np.any(np.diff(wvl_FFT_trace) < 0):
            if np.all(np.diff(wvl_FFT_trace) < 0):
                wvl_FFT_trace = wvl_FFT_trace[::-1]
                spectrum_FFT_trace = spectrum_FFT_trace[::-1]
                phase_FFT_trace = phase_FFT_trace[::-1]
            else:
                raise ValueError(
                    "in function TraceHandler.load_from_spectral_data() wavelength array is not monotonous"
                )

        # frequency spectrum (monotonically increasing)
        freq = constants.speed_of_light / wvl_FFT_trace[::-1] * 1e-6
        spectrum_freq = spectrum_FFT_trace[::-1] / (
            constants.speed_of_light / wvl_FFT_trace[::-1] ** 2 * 1e-6
        )
        phase_FFT_trace = phase_FFT_trace[::-1]

        # create proper freq axis for the fourier transform
        df = (freq[1] - freq[0]) / 4
        nf = int(np.ceil(3 * freq[-1] / df))
        self.frequencyAxis = np.concatenate((np.arange(0, nf), np.arange(-nf, 0))) * df
        # corresponding time window
        twin = 1.0 / (df)

        # complex fourier transform
        spectrum_freq = np.sqrt(spectrum_freq) * np.exp(1.0j * phase_FFT_trace)

        # fill with zeros
        initial_zero_freq = np.linspace(
            freq[1] - freq[0], freq[0], int(np.ceil(4 * freq[0] / (freq[1] - freq[0])))
        )
        initial_zeros = np.zeros(len(initial_zero_freq))
        final_zero_freq = np.linspace(
            2 * freq[-1] - freq[-2],
            4 * freq[-1],
            int(np.ceil(15 * freq[-1] / (freq[-1] - freq[-2]))),
        )
        final_zeros = np.zeros(len(final_zero_freq))
        freq = np.concatenate((initial_zero_freq, freq, final_zero_freq))
        spectrum_freq = np.concatenate((initial_zeros, spectrum_freq, final_zeros))
        # add negative frequencies
        freq = np.concatenate((-freq[::-1], freq))
        spectrum_freq = np.concatenate(
            (np.conjugate(spectrum_freq[::-1]), spectrum_freq)
        )

        # interpolate the spectrum to the frequency axis of the fft
        self.fftFieldV = np.interp(self.frequencyAxis, freq, spectrum_freq)

        # add linear phase to center the pulse in the time window; this has to be done after interpolation because the spectral phase will be fast oscillating
        self.fftFieldV = self.fftFieldV * np.exp(
            1.0j * 2 * np.pi * self.frequencyAxis * twin / 2
        )

        self.update_fft_spectrum()
        self.update_trace_from_fft()

    def load_spectrum(self, fname=None):
        """Loads spectrum from file.

        The spectrum does not correspond to the fourier transform of the trace, but to the spectrometer data that you would like to compare with the loaded trace

        Args:
            fname: filename

        """
        if fname is not None:
            self.filename_spectrum = fname
        data = pd.read_csv(self.filename_spectrum, sep="\t")
        self.wvlSpectrometer = data["wavelength (nm)"].to_numpy()
        self.ISpectrometer = data["intensity (a.u.)"].to_numpy()
        self.normalize_spectrum()

    def update_fft(self, zero_pad_field=True):
        """Updates the fft of the trace from the time domain data.

        Args:
            zero_pad_field: bool; if true (default) add zeros before and after the trace before computing fft (for a smoother spectrum)
                The length of appended zeros is defined by self.fsZeroPadding.

        """
        zp_fieldTimeV, zp_fieldV = deepcopy(self.fieldTimeV), deepcopy(self.fieldV)
        if zero_pad_field:
            zp_fieldTimeV, zp_fieldV = zero_padding(
                self.fieldTimeV, self.fieldV, fsExtension=self.fsZeroPadding
            )
        self.frequencyAxis, self.fftFieldV = fourier_transform(zp_fieldTimeV, zp_fieldV)
        self.update_fft_spectrum()

    def update_fft_spectrum(self):
        """Updates the wavelength-dependent intensity spectrum. Results are stored in self.wvlAxis and self.fftSpectrum.

        This is (derived from but) different from the fourier transform because:
         * it considers the squared modulus of the field
         * it multiplies the frequency-dependent spectral density by df/dλ = c/λ²
         * it considers only positive wavelengths

        The function internally calls update_spectral_phase() as well.
        """
        self.wvlAxis = (
            constants.speed_of_light
            / self.frequencyAxis[self.frequencyAxis > 0]
            * 1.0e-6
        )
        self.fftSpectrum = (
            np.abs(self.fftFieldV[self.frequencyAxis > 0]) ** 2
            * constants.speed_of_light
            / self.wvlAxis**2
        )
        self.compute_complex_field()
        self.update_spectral_phase()
        self.normalize_fft_spectrum()

    def update_spectral_phase(self):
        """Computes the spectral phase of the trace. The result is stored in self.fftphase.

        An attempt of automatic linear phase subtraction is made. Phase is unwrapped.
        """
        # determine the time distance (delay) of the waveform peak from the beginning of the trace (more precisely, from the beginning of the ifft)
        tmax = self.get_zero_delay()
        deltaT1 = tmax - self.complexFieldTimeV[0]
        deltaT2 = self.complexFieldTimeV[-1] - tmax
        ifft_twindow = 1 / (self.frequencyAxis[1] - self.frequencyAxis[0])
        total_delay = (
            ifft_twindow / 2 + (deltaT2 - deltaT1) / 2
        )  # total estimated delay of the peak from the beginning of the ifft trace

        # compute the phase, use the computed 'delay' to correct for the non-interesting linear component, and unwrap it
        self.fftphase = np.angle(self.fftFieldV[self.frequencyAxis > 0])
        self.fftphase = (
            self.fftphase
            - total_delay * self.frequencyAxis[self.frequencyAxis > 0] * 2 * np.pi
        )
        self.fftphase = np.unwrap(self.fftphase)

        # subtract constant phase offset (m * 2*π)
        n_pi = np.sum(self.fftSpectrum * self.fftphase) / np.sum(self.fftSpectrum)
        n_pi = int(n_pi / (2 * np.pi))
        self.fftphase -= n_pi * 2 * np.pi

    def update_trace_from_fft(self):  # remember to use together with strip_from_trace
        # careful: the complex part of the IFFT is discarded
        """Updates the field trace from fft.

        In most cases, you might want to call strip_from_trace() afterward, to eliminate the zero-padding from the trace.
        """
        self.fieldTimeV, self.fieldV = inverse_fourier_transform(
            self.frequencyAxis, self.fftFieldV
        )
        self.fieldV = np.real(self.fieldV)
        self.normalization_trace = np.max(np.abs(self.fieldV))
        self.get_zero_delay()

    def strip_from_trace(self, timeRange=None):
        """Eliminates the zeros appended to the trace (zero-padding) when computing the FFT.

        Useful when retrieving the trace via ifft (method update_trace_from_fft()).

        Args:
            timeRange (float): the time range (fs) to strip from the beginning and end of the trace. If None (default), fsZeroPadding is used.

        """
        if timeRange is None:
            timeRange = self.fsZeroPadding
        # strip the first and last fsZeroPadding time points
        self.fieldV = self.fieldV[
            (self.fieldTimeV >= np.min(self.fieldTimeV) + timeRange)
            & (self.fieldTimeV <= np.max(self.fieldTimeV) - timeRange)
        ]
        self.fieldTimeV = self.fieldTimeV[
            (self.fieldTimeV >= np.min(self.fieldTimeV) + timeRange)
            & (self.fieldTimeV <= np.max(self.fieldTimeV) - timeRange)
        ]
        if self.fieldStdevV is None:
            return True
        if len(self.fieldStdevV) > len(self.fieldV):
            n_remove = len(self.fieldStdevV) - len(self.fieldV)
            print(
                "\n\nWarning: fieldStdevV is longer than fieldV, removing",
                n_remove,
                " elements\n\n",
            )
            while n_remove > 0:
                if n_remove % 2 == 0:
                    self.fieldStdevV = np.delete(self.fieldStdevV, 0)
                else:
                    self.fieldStdevV = np.delete(self.fieldStdevV, -1)
                n_remove -= 1
        elif len(self.fieldStdevV) < len(self.fieldV):
            n_add = len(self.fieldV) - len(self.fieldStdevV)
            print(
                "\n\nWarning: fieldStdevV is shorter than fieldV, adding",
                n_add,
                " elements\n\n",
            )
            while n_add > 0:
                if n_add % 2 == 0:
                    self.fieldStdevV = np.insert(self.fieldStdevV, 0, 0)
                else:
                    self.fieldStdevV = np.append(self.fieldStdevV, 0)
                n_add -= 1
        return True

    def strip_from_complex_trace(self, timeRange=None):
        """From the complex trace eliminates the zeros appended (zero-padding) when computing the FFT (Analogous to strip_from_trace())."""
        if timeRange is None:
            timeRange = self.fsZeroPadding
        # strip the first and last fsZeroPadding time points
        self.complexFieldV = self.complexFieldV[
            (self.complexFieldTimeV >= np.min(self.complexFieldTimeV) + timeRange)
            & (self.complexFieldTimeV <= np.max(self.complexFieldTimeV) - timeRange)
        ]
        self.complexFieldTimeV = self.complexFieldTimeV[
            (self.complexFieldTimeV >= np.min(self.complexFieldTimeV) + timeRange)
            & (self.complexFieldTimeV <= np.max(self.complexFieldTimeV) - timeRange)
        ]

    def tukey_time_window(self, lowEdge, upEdge, lowEdgeWidth, upEdgeWidth):
        """Applies a tukey window to the trace in the time domain.

        Args:
            lowEdge, upEdge: lower and upper edge of the window upEdge-lowEdge = FWHM of the window
            lowEdgeWidth, upEdgeWidth: width of the cosine-shaped edges (from 0 to 1 or viceversa)

        """
        window = asymmetric_tukey_window(
            np.abs(self.fieldTimeV), lowEdge, upEdge, lowEdgeWidth, upEdgeWidth
        )
        self.fieldV = self.fieldV * window
        self.update_fft()

    def fourier_interpolation(self, ntimes_finer: int):
        """Shrinks the time step of the trace in time domain by a factor ntimes_finer.

        This is done by 'zero-padding' the fourier transform of the trace, i.e., by extending the frequency axis by a factor ntimes_finer, and adding corresponding zeros to the FFT field.

        Args:
            ntimes_finer (int): the factor by which to shrink the time step of the trace in time domain. Must be >= 1.

        """
        if ntimes_finer < 1:
            raise ValueError(
                "in function TraceHandler.fourier_interpolation() ntimes_finer must be >= 1"
            )
        if self.frequencyAxis is None or self.fftFieldV is None:
            raise ValueError(
                "in function TraceHandler.fourier_interpolation() frequency axis or fft field is not defined"
            )
        # extend the frequency axis by a factor ntimes_finer
        df = self.frequencyAxis[1] - self.frequencyAxis[0]
        i_last_pos = int(np.ceil(len(self.frequencyAxis) / 2) - 1)
        if self.frequencyAxis[i_last_pos] < 0 or self.frequencyAxis[i_last_pos + 1] > 0:
            print(self.frequencyAxis[0 : i_last_pos + 1])
            print(self.frequencyAxis[i_last_pos + 1 :])
            raise Exception(
                "in function fourier_interpolation() frequency axes was not extended correctly"
            )

        appended_freqFFT = np.linspace(
            self.frequencyAxis[i_last_pos] + df,
            ntimes_finer * self.frequencyAxis[i_last_pos] + df,
            round((ntimes_finer - 1) * self.frequencyAxis[i_last_pos] / df),
        )
        prepended_freqFFT = np.linspace(
            -(ntimes_finer - 1) * self.frequencyAxis[i_last_pos]
            + self.frequencyAxis[i_last_pos + 1],
            +self.frequencyAxis[i_last_pos + 1],
            round((ntimes_finer - 1) * self.frequencyAxis[i_last_pos] / df),
        )
        self.frequencyAxis = np.concatenate(
            (
                self.frequencyAxis[: i_last_pos + 1],
                appended_freqFFT,
                prepended_freqFFT,
                self.frequencyAxis[i_last_pos + 1 :],
            )
        )
        self.fftFieldV = np.concatenate(
            (
                self.fftFieldV[: i_last_pos + 1],
                np.zeros(len(appended_freqFFT)),
                np.zeros(len(prepended_freqFFT)),
                self.fftFieldV[i_last_pos + 1 :],
            )
        )

        self.update_trace_from_fft()
        self.strip_from_trace()
        self.update_fft_spectrum()

    def differentiate_trace(self, spectrally=True):
        """Computes the time derivative of the trace.

        Args:
            spectrally: bool; if True (default) the derivative will be computed in the spectral domain (maybe more stable numerically)

        """
        if spectrally:
            self.fftFieldV = 1j * self.frequencyAxis * self.fftFieldV
            self.update_trace_from_fft()
            self.update_fft_spectrum()
        else:
            self.fieldV = np.gradient(self.fieldV, self.fieldTimeV)
            self.update_fft()

    def integrate_trace(self, spectrally=True):
        """Integrates the trace in time.

        Args:
            spectrally (bool): if True (default) the integral will be computed in the spectral domain (maybe more stable numerically)

        """
        if spectrally:
            if self.frequencyAxis[0] != 0:
                print(
                    "WARNING: in function TraceHandler.integrate_trace() the FFT axis' first element is not zero"
                )
            self.fftFieldV[1:] = -1j / self.frequencyAxis[1:] * self.fftFieldV[1:]
            self.update_trace_from_fft()
            self.update_fft_spectrum()
        else:
            self.fieldV = np.cumsum(self.fieldV) * np.mean(np.diff(self.fieldTimeV))
            self.update_fft()

    def save_trace_to_file(self, filename, low_lim=None, up_lim=None):
        """Saves the trace to file.

        Args:
            filename
            low_lim, up_lim: only save the trace between low_lim and up_lim (default: None, None)

        """
        if low_lim is not None and up_lim is not None:
            fieldTimeV_to_write = self.fieldTimeV[
                (self.fieldTimeV >= low_lim) & (self.fieldTimeV <= up_lim)
            ]
            fieldV_to_write = self.fieldV[
                (self.fieldTimeV >= low_lim) & (self.fieldTimeV <= up_lim)
            ]
        else:
            fieldTimeV_to_write = self.fieldTimeV
            fieldV_to_write = self.fieldV
        data = pd.DataFrame(
            {"delay (fs)": fieldTimeV_to_write, "field (a.u.)": fieldV_to_write}
        )
        data.to_csv(filename, sep="\t", index=False)

    def normalize_spectrum(self):
        """Normalizes the comparison spectrum to its integral."""
        spectrum_range = [60, 930]
        n_factor = (
            np.abs(
                np.diff(self.wvlSpectrometer)[
                    (self.wvlSpectrometer[:-1] > spectrum_range[0])
                    & (self.wvlSpectrometer[:-1] < spectrum_range[1])
                ]
            )
            * self.ISpectrometer[:-1][
                (self.wvlSpectrometer[:-1] > spectrum_range[0])
                & (self.wvlSpectrometer[:-1] < spectrum_range[1])
            ]
        ).sum()
        self.ISpectrometer /= n_factor

    def normalize_fft_spectrum(self, spectrum_range=[60, 930]):
        """Normalizes the spectrum of the trace (|FFT{trace}|^2 * df/dλ) to its integral (area).

        Args:
            spectrum_range (list): the range of wavelengths (nm) to consider for the normalization. Default is [60, 930] nm.

        """
        n_factor = (
            np.abs(
                np.diff(self.wvlAxis)[
                    (self.wvlAxis[:-1] > spectrum_range[0])
                    & (self.wvlAxis[:-1] < spectrum_range[1])
                ]
            )
            * self.fftSpectrum[:-1][
                (self.wvlAxis[:-1] > spectrum_range[0])
                & (self.wvlAxis[:-1] < spectrum_range[1])
            ]
        ).sum()
        self.fftSpectrum /= n_factor

    def get_trace(self):
        """Returns the time (fs) and field arrays. Field is in a.u. unless set_fluence() was called or a calibrated field was given as an input."""
        return self.fieldTimeV, self.fieldV

    def get_spectrum_trace(self):
        """Returns the wavelength and spectral intensity corresponding to the fourier transform of the trace."""
        return self.wvlAxis, self.fftSpectrum

    def get_spectral_phase(self):
        """Returns the wavelength array and the spectral phase array corresponding to the FFT of the trace."""
        return self.wvlAxis, self.fftphase

    def get_stdev(self):
        """Returns only the standard deviation (sigma) array."""
        return self.fieldStdevV

    def get_positive_fft_field(self):
        """Returns the positive frequency axis f and the corresponding FFT field (complex).

        Main task of this function is to 'smooth' the phase of the fft (= remove linear phase, corresponding to a time shift in the temporal domain), since the waveform is usually centered around t = 0
        """
        if self.frequencyAxis is None or self.fftFieldV is None:
            raise ValueError(
                "in function TraceHandler.get_positive_fft_field() frequency axis or fft field is not defined"
            )
        positive_freq = self.frequencyAxis[self.frequencyAxis >= 0]
        positive_fft_field = self.fftFieldV[self.frequencyAxis >= 0]
        # remove linear phase
        tmax = self.get_zero_delay()
        deltaT1 = tmax - self.fieldTimeV[0]
        deltaT2 = self.fieldTimeV[-1] - tmax
        ifft_twindow = 1 / (self.frequencyAxis[1] - self.frequencyAxis[0])
        total_delay = ifft_twindow / 2 + (deltaT2 - deltaT1) / 2
        positive_fft_field = positive_fft_field * np.exp(
            -1.0j * 2 * np.pi * positive_freq * total_delay
        )
        return positive_freq, positive_fft_field

    def get_fluence(self):
        """Calculates the fluence of the trace.

        convention: field [V/Å], time [fs], fluence [J/cm²]
        F = c eps_0 integral(E^2)dt


        Returns:
            fluence: float

        """
        return (
            constants.speed_of_light
            * constants.epsilon_0
            * np.trapz(self.fieldV**2, self.fieldTimeV)
            * 1e1
        )

    def set_fluence(self, fluence):
        """Set the fluence of the trace.

        Convention: field [V/Å], time [fs], fluence [J/cm²]

        F = c eps_0 integral(E^2)dt
        """
        F = (
            constants.speed_of_light
            * constants.epsilon_0
            * np.trapz(self.fieldV**2, self.fieldTimeV)
            * 1e1
        )
        self.fieldV *= np.sqrt(fluence / F)
        self.update_fft()
        self.update_fft_spectrum()

    def compute_complex_field(self):
        """Computes the complex trace by inverting the positive-freq FFT and stores it in self.complexFieldV (for envelope or phase computation, for example).

        Notice that the fourier transform of the trace is not computed, it is assumed to be already stored in the class. This should be true in most cases.
        """
        complex_spectrum = deepcopy(self.fftFieldV)
        complex_spectrum[self.frequencyAxis < 0] = 0
        self.complexFieldTimeV, self.complexFieldV = inverse_fourier_transform(
            self.frequencyAxis, complex_spectrum
        )
        self.complexFieldV = 2 * self.complexFieldV
        self.strip_from_complex_trace()

    def get_envelope(self):
        """Get the envelope of the trace.

        Returns:
            time array (fs)
            envelope array

        """
        if self.complexFieldV is None:
            self.compute_complex_field()
        return self.complexFieldTimeV, np.abs(self.complexFieldV)

    def get_phase(self):
        """Get the instantaneous phase of the trace.

        If complexFieldV is already computed and stored, no re-calculation occurs

        Returns:
            time array (fs)
            instantaneous phase array (fs)

        """
        if self.complexFieldV is None:
            self.compute_complex_field()
        return self.complexFieldTimeV, np.angle(self.complexFieldV)

    def get_zero_delay(self):
        """Get the time value corresponding to the envelope peak.

        Returns:
            zero_delay: float

        """
        # careful: the time grid should be fine enough to resolve the maximum of the envelope
        t, en = self.get_envelope()
        dt = t[1] - t[0]
        max_index, max_value = find_maximum_location(en)
        self.zero_delay = t[0] + dt * max_index
        return self.zero_delay

    def get_FWHM(self):
        """Get the FWHM of the trace.

        Returns:
            FWHM: float

        """
        t, en = self.get_envelope()
        dt = t[1] - t[0]
        return fwhm(en**2, dt)

    def fft_tukey_bandpass(
        self, lowWavelengthEdge, upWavelengthEdge, lowEdgeWidth, highEdgeWidth
    ):
        """Applies a bandpass filter to the trace in the frequency domain using a tukey window.

        The tukey window is 1 between  lowWavelengthEdge+lowEdgeWidth/2  and  upWavelengthEdge-highEdgeWidth/2
        and it is reaches zero at lowWavelengthEdge-lowEdgeWidth/2  and  upWavelengthEdge+highEdgeWidth/2.
        Notice that the edges are only cosine-shaped in the frequency domain. In the wavelength domain upWavelengthEdge - lowWavelengthEdge does not coincide with the FWHM of the tukey function

        Args:
            lowWavelengthEdge: float (nm)
            upWavelengthEdge: float (nm)
            lowEdgeWidth: float (nm)
            highEdgeWidth: float (nm)

        """
        upFreqEdge = (
            constants.speed_of_light / (lowWavelengthEdge - lowEdgeWidth / 2) / 2 * 1e-6
            + constants.speed_of_light
            / (lowWavelengthEdge + lowEdgeWidth / 2)
            / 2
            * 1e-6
        )
        lowFreqEdge = (
            constants.speed_of_light / (upWavelengthEdge - highEdgeWidth / 2) / 2 * 1e-6
            + constants.speed_of_light
            / (upWavelengthEdge + highEdgeWidth / 2)
            / 2
            * 1e-6
        )

        upFreqEdgeWidth = (
            constants.speed_of_light / (lowWavelengthEdge - lowEdgeWidth / 2) * 1e-6
            - constants.speed_of_light / (lowWavelengthEdge + lowEdgeWidth / 2) * 1e-6
        )
        lowFreqEdgeWidth = (
            constants.speed_of_light / (upWavelengthEdge - highEdgeWidth / 2) * 1e-6
            - constants.speed_of_light / (upWavelengthEdge + highEdgeWidth / 2) * 1e-6
        )

        window = asymmetric_tukey_window(
            np.abs(self.frequencyAxis),
            lowFreqEdge,
            upFreqEdge,
            lowFreqEdgeWidth,
            upFreqEdgeWidth,
        )
        self.fftFieldV = self.fftFieldV * window
        self.update_trace_from_fft()
        self.update_fft_spectrum()
        self.strip_from_trace()

    def apply_transmission(self, wavelengths, f):
        """Applies a spectral transmission function to the trace (e.g. spectral filter).

        Args:
            wavelengths: ndarray = wavelength array (nm)
            f: ndarray = transmission function f(λ)

        """
        if np.any(np.diff(wavelengths)) < 0:
            if np.all(np.diff(wavelengths)) < 0:
                wavelengths = wavelengths[::-1]
                f = f[::-1]
            else:
                raise ValueError(
                    "in function TraceHandler.apply_transmission() wavelength array is not monotonous"
                )
        freq = constants.speed_of_light / wavelengths[::-1] * 1e-6
        spectrum_freq = f[::-1]

        # fill with zeros
        initial_zero_freq = np.linspace(
            freq[1] - freq[0], freq[0], int(np.ceil(4 * freq[0] / (freq[1] - freq[0])))
        )
        initial_zeros = np.zeros(len(initial_zero_freq))
        final_zero_freq = np.linspace(
            2 * freq[-1] - freq[-2],
            4 * freq[-1],
            int(np.ceil(15 * freq[-1] / (freq[-1] - freq[-2]))),
        )
        final_zeros = np.zeros(len(final_zero_freq))
        freq = np.concatenate((initial_zero_freq, freq, final_zero_freq))
        spectrum_freq = np.concatenate((initial_zeros, spectrum_freq, final_zeros))
        # add negative frequencies
        freq = np.concatenate((-freq[::-1], freq))
        spectrum_freq = np.concatenate((spectrum_freq[::-1], spectrum_freq))

        # interpolate the spectrum to the frequency axis of the fft
        spectrum_interp = np.interp(self.frequencyAxis, freq, spectrum_freq)

        self.fftFieldV = self.fftFieldV * spectrum_interp
        self.update_trace_from_fft()
        self.update_fft_spectrum()
        self.strip_from_trace()

    def apply_spectrum(
        self,
        wvl=None,
        spectrum=None,
        CEP_shift: float = 0.0,
        stripZeroPadding: bool = True,
    ):
        """Applies a spectrum to the phase of the trace. This means that a new trace is computed and stored in the TraceHandler object (replacing the existing one);
        the new trace has spectral intensity equal to the applied spectrum and spectral phase equal to the phase of the existing trace.

        Args:
            wvl: wavelength array (nm). If None (default) the comparison spectrum stored in the class (self.wvlSpectrometer and self.ISpectrometer) is applied.
            spectrum: spectral intensity array. If None (default) the comparison spectrum stored in the class (self.wvlSpectrometer and self.ISpectrometer) is applied.
            CEP_shift: a possible artificial phase shift IN UNITS OF PI! (default = 0)
            stripZeroPadding (bool): whether to eliminate the zero padding (zeros appended to the trace). Default is True

        """
        if wvl is None or spectrum is None:
            wvl = self.wvlSpectrometer
            spectrum = self.ISpectrometer
        # check that the wvl array is monotonically increasing
        if np.any(np.diff(wvl) < 0):
            if np.all(np.diff(wvl) < 0):
                wvl = wvl[::-1]
                spectrum = spectrum[::-1]
            else:
                raise ValueError(
                    "in function TraceHandler.apply_spectrum() wavelength array is not monotonous"
                )

        # frequency spectrum (monotonically increasing)
        freq = constants.speed_of_light / wvl[::-1] * 1e-6
        spectrum_freq = spectrum[::-1] / (
            constants.speed_of_light / wvl[::-1] ** 2 * 1e-6
        )
        spectrum_freq = np.maximum(spectrum_freq, np.zeros(len(spectrum_freq)))

        # fill with zeros
        initial_zero_freq = np.linspace(
            freq[1] - freq[0], freq[0], int(np.ceil(4 * freq[0] / (freq[1] - freq[0])))
        )
        initial_zeros = np.zeros(len(initial_zero_freq))
        final_zero_freq = np.linspace(
            2 * freq[-1] - freq[-2],
            4 * freq[-1],
            int(np.ceil(15 * freq[-1] / (freq[-1] - freq[-2]))),
        )
        final_zeros = np.zeros(len(final_zero_freq))
        freq = np.concatenate((initial_zero_freq, freq, final_zero_freq))
        spectrum_freq = np.concatenate((initial_zeros, spectrum_freq, final_zeros))
        # add negative frequencies
        freq = np.concatenate((-freq[::-1], freq))
        spectrum_freq = np.concatenate((spectrum_freq[::-1], spectrum_freq))

        # interpolate the spectrum to the frequency axis of the fft
        spectrum_interp = np.interp(self.frequencyAxis, freq, spectrum_freq)
        self.fftFieldV = np.sqrt(spectrum_interp) * np.exp(
            1j * np.angle(self.fftFieldV)
            + 1j * np.pi * CEP_shift * np.sign(self.frequencyAxis)
        )

        self.update_trace_from_fft()
        if stripZeroPadding:
            self.strip_from_trace()
        else:
            self.fieldStdevV = None
        self.update_fft_spectrum()

    def fresnel_reflection(
        self,
        material2,
        angle_in,
        material1=None,
        forward: bool = True,
        s_polarized: bool = True,
        path: str = "./RefractiveIndices/",
    ):
        """Calculates the fresnel reflection of the pulse at the interface between two materials. The waveform is travelling from material 1 to material 2.
        The first medium (material1) should be non-absorptive. As usual the resulting waveform is stored in the TraceHandler object, replacing the previous one.

        Refractive index files should contain 3 space-separated columns, respectively with headers: wvl n k, where wvl is the wavelength in um, n and k resp. the real and imaginary part of the refractive index.

        Args:
            material2: the filename (without '.txt') of the refractive index data for the material after the interface (e.g. Si Al MgF2); wavelength is in um in the refractive index file
            angle_in: the incidence angle in degrees
            material1: the filename (without '.txt') of the refractive index data for the material before the interface. If None (default), vacuum is assumed
            forward (bool):  if True (default) forward reflection is computed (the result waveform is the reflection of the previously stored waveform.

                if False backward reflection is computed (the previous waveform is the reflection of the result waveform)
            s_polarized (bool): True. Reflection calculation only implemented for s-polarized light
            path (str): path for the refractive index files. Defaults to "./RefractiveIndices/"

        """
        if not s_polarized:
            raise ValueError(
                "in function TraceHandler.fresnel_reflection() p_polarized is not implemented yet\n"
            )

        # read refractive index
        refIndexData = pd.read_table(
            path + material2 + ".txt", sep=" ", keep_default_na=False
        )
        wvl2 = np.array(refIndexData["wvl"]) * 1e3
        n2 = np.array(refIndexData["n"]) + 1j * np.array(refIndexData["k"])
        if material1 is None:
            wvl1 = np.array(wvl2)
            n1 = n2 * 0 + 1
        else:
            refIndexData = pd.read_table(
                path + material1 + ".txt", sep=" ", keep_default_na=False
            )
            wvl1 = np.array(refIndexData["wvl"]) * 1e3
            n1 = np.array(refIndexData["n"]) + 1j * np.array(refIndexData["k"])

        # check that the wvl arrays are monotonically increasing
        if np.any(np.diff(wvl1) < 0):
            if np.all(np.diff(wvl1) < 0):
                wvl1 = wvl1[::-1]
                n1 = n1[::-1]
            else:
                raise ValueError(
                    "in function TraceHandler.fresnel_reflection() wavelength array is not monotonous"
                )
        if np.any(np.diff(wvl2) < 0):
            if np.all(np.diff(wvl2) < 0):
                wvl2 = wvl2[::-1]
                n2 = n2[::-1]
            else:
                raise ValueError(
                    "in function TraceHandler.fresnel_reflection() wavelength array is not monotonous"
                )

        # frequency spectrum (monotonically increasing)
        freq1 = constants.speed_of_light / wvl1[::-1] * 1e-6
        freq2 = constants.speed_of_light / wvl2[::-1] * 1e-6
        n1 = n1[::-1]
        n2 = n2[::-1]

        # fill with ones
        initial_ones_freq = np.linspace(
            freq1[1] - freq1[0],
            freq1[0],
            int(np.ceil(4 * freq1[0] / (freq1[1] - freq1[0]))),
        )
        initial_ones = np.ones(len(initial_ones_freq)) * n1[0]
        final_ones_freq = np.linspace(
            2 * freq1[-1] - freq1[-2],
            4 * freq1[-1],
            int(np.ceil(15 * freq1[-1] / (freq1[-1] - freq1[-2]))),
        )
        final_ones = np.ones(len(final_ones_freq)) * n1[-1]
        freq1 = np.concatenate((initial_ones_freq, freq1, final_ones_freq))
        n1 = np.concatenate((initial_ones, n1, final_ones))
        # same for n2
        initial_ones_freq = np.linspace(
            freq2[1] - freq2[0],
            freq2[0],
            int(np.ceil(4 * freq2[0] / (freq2[1] - freq2[0]))),
        )
        initial_ones = np.ones(len(initial_ones_freq)) * n2[0]
        final_ones_freq = np.linspace(
            2 * freq2[-1] - freq2[-2],
            4 * freq2[-1],
            int(np.ceil(15 * freq2[-1] / (freq2[-1] - freq2[-2]))),
        )
        final_ones = np.ones(len(final_ones_freq)) * n2[-1]
        freq2 = np.concatenate((initial_ones_freq, freq2, final_ones_freq))
        n2 = np.concatenate((initial_ones, n2, final_ones))

        # add negative frequencies
        freq1 = np.concatenate((-freq1[::-1], freq1))
        n1 = np.concatenate((np.conjugate(n1[::-1]), n1))
        freq2 = np.concatenate((-freq2[::-1], freq2))
        n2 = np.concatenate((np.conjugate(n2[::-1]), n2))

        # interpolate the n1 and n2 to the frequency axis of the fft
        n1Interp = np.interp(self.frequencyAxis, freq1, n1)
        n2Interp = np.interp(self.frequencyAxis, freq2, n2)

        # calculate the fresnel reflection coefficients using only the angle of incidence (NOT SURE THIS WORKS WHEN THE FIRST MEDIUM IS LOSSY)
        r = (
            n1Interp * np.cos(angle_in * np.pi / 180)
            - n2Interp
            * np.sqrt(1 - (n1Interp / n2Interp * np.sin(angle_in * np.pi / 180)) ** 2)
        ) / (
            n1Interp * np.cos(angle_in * np.pi / 180)
            + n2Interp
            * np.sqrt(1 - (n1Interp / n2Interp * np.sin(angle_in * np.pi / 180)) ** 2)
        )
        if forward:
            self.fftFieldV = self.fftFieldV * r
        else:
            self.fftFieldV = self.fftFieldV / r

        self.fieldStdevV = None

        self.update_trace_from_fft()
        self.strip_from_trace()
        self.update_fft_spectrum()

    def apply_zero_phase(self):
        """Applies zero-phase to the trace; this allows, for example, to retrieve the fourier limited pulse corresponding to the same FFT spectrum of the loaded trace."""
        tau = (
            self.fsZeroPadding + (np.max(self.fieldTimeV) - np.min(self.fieldTimeV)) / 2
        )
        self.fftFieldV = np.abs(self.fftFieldV) * np.exp(
            1.0j * (2 * np.pi * tau * self.frequencyAxis)
        )
        self.update_trace_from_fft()
        self.strip_from_trace()
        self.update_fft_spectrum()

    def time_frequency_analysis(
        self, sigma_time, low_lim=None, up_lim=None, low_lim_freq=None, up_lim_freq=None
    ):
        """Performs time-frequency analysis by using scipy's short time fourier transform (fourier transform of the trace convoluted by a 'sigma_time' broad gaussian).

        Args:
            sigma_time: sigma of the gaussian window
            low_lim, up_lim (float): xaxis limits for plotting. Default None
            low_lim_freq, up_lim_freq (float): xaxis limits for plotting. Default None

        """
        dt = np.mean(np.diff(self.fieldTimeV))
        w = scipy.signal.windows.gaussian(
            int(sigma_time / dt * 6) + 1, sigma_time / dt, sym=True
        )
        TFA = scipy.signal.ShortTimeFFT(
            w, hop=1, fs=1.0 / dt, mfft=int(sigma_time / dt * 24), scale_to="magnitude"
        )
        TFData = TFA.stft(self.fieldV)

        fig, ax = plt.subplots()
        t_lo, t_hi, f_lo, f_hi = TFA.extent(
            self.fieldV.size
        )  # time and freq range of plot
        if low_lim is None:
            low_lim = t_lo + self.fieldTimeV[0]
        if up_lim is None:
            up_lim = t_hi + self.fieldTimeV[0]
        if low_lim_freq is None:
            low_lim_freq = 0
        if up_lim_freq is None:
            up_lim_freq = 3.0

        ax.set(xlim=(low_lim, up_lim), ylim=(low_lim_freq, up_lim_freq))
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Frequency (PHz)")
        im1 = ax.imshow(
            abs(TFData) / np.max(abs(TFData)),
            origin="lower",
            aspect="auto",
            extent=(t_lo + self.fieldTimeV[0], t_hi + self.fieldTimeV[0], f_lo, f_hi),
            cmap="viridis",
        )
        cbar = fig.colorbar(im1)

        cbar.ax.set_ylabel("Magnitude of the field (Arb. unit)")
        fig.tight_layout()
        return TFData

    def plot_trace(self, low_lim=None, up_lim=None, normalize: bool = True):
        """Plots the field trace.

        Args:
            low_lim, up_lim: float = xaxis limits for plotting. Default None
            normalize: bool = if True (default) normalize the peak of the trace to 1

        """
        fig, ax = plt.subplots()
        if normalize:
            norm_plot = self.normalization_trace
        else:
            norm_plot = 1
        main_line = ax.plot(
            self.fieldTimeV, self.fieldV / norm_plot, label="Field trace"
        )
        if self.fieldStdevV is not None:
            ax.fill_between(
                self.fieldTimeV,
                (self.fieldV - self.fieldStdevV) / norm_plot,
                (self.fieldV + self.fieldStdevV) / norm_plot,
                color=main_line[0].get_color(),
                alpha=0.3,
            )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Field (Arb. unit)")

        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim, up_lim)
        return fig

    def plot_spectrum(
        self, low_lim=40, up_lim=1000, no_phase: bool = False, phase_blanking_level=0.05
    ):
        """Plots the trace spectrum and phase together with the spectrometer measurement [if provided].

        Args:
            low_lim, up_lim (float): xaxis limits for plotting. Default: 40, 1000
            no_phase: if True, don't plot the phase. Default: False

        """
        fig, ax = plt.subplots()

        if not no_phase:
            ax2 = ax.twinx()
        lines = []
        min_intensity = phase_blanking_level * np.max(self.fftSpectrum)
        lines += ax.plot(
            self.wvlAxis[(self.wvlAxis > low_lim) & (self.wvlAxis < up_lim)],
            self.fftSpectrum[(self.wvlAxis > low_lim) & (self.wvlAxis < up_lim)],
            label="Fourier transform",
        )
        if not no_phase:
            ax2.plot([], [])
            if self.wvlSpectrometer is not None:
                ax2.plot([], [])
            lines += ax2.plot(
                self.wvlAxis[
                    (self.wvlAxis > low_lim)
                    & (self.wvlAxis < up_lim)
                    & (self.fftSpectrum > min_intensity)
                ],
                self.fftphase[
                    (self.wvlAxis > low_lim)
                    & (self.wvlAxis < up_lim)
                    & (self.fftSpectrum > min_intensity)
                ],
                "--",
                label="Phase",
            )
        if self.wvlSpectrometer is not None:
            lines += ax.plot(
                self.wvlSpectrometer[
                    (self.wvlSpectrometer > low_lim) & (self.wvlSpectrometer < up_lim)
                ],
                self.ISpectrometer[
                    (self.wvlSpectrometer > low_lim) & (self.wvlSpectrometer < up_lim)
                ],
                label="Spectrometer",
            )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        ax.tick_params(axis="both")
        if not no_phase:
            ax2.set_ylabel("Phase (rad)")
            ax2.tick_params(axis="y")
        ax.legend(lines, [line.get_label() for line in lines])
        return fig


class MultiTraceHandler:
    """Initializes and stores multiple TraceHandler objects. It is used to plot multiple traces in a single graph, compare them, and perform operations between them.

    ALL TIMES IN FS, ALL WAVELENGTHS IN NM, ALL FREQUENCIES IN PHz

    All stored traces are accessible as TraceHandler object in the corresponding list.

    Attributes:
        traceHandlers (list): list of traceHandler objects
        fsZeroPadding (float): zero padding in fs for the traces (default 150 fs)

    """

    def __init__(
        self,
        filenameList=None,
        filenameSpectrumList=None,
        timeList=None,
        fieldList=None,
        stdevList=None,
        wvlList=None,
        spectrumList=None,
        traceHandlerList=None,
    ):
        """Constructor of MultiTraceHandler.

        One of the following arguments must be provided (all arguments default to None:
        * filenameList: list of filenames containing the field traces (see TraceHandler constructor)
        * timeList AND fieldList: list of time arrays and field arrays (see TraceHandler constructor)
        * traceHandlerList: list of TraceHandler objects

        Args:
            filenameList : list of filenames containing the field traces (see TraceHandler constructor)
            filenameSpectrumList : list of filenames containing the spectrum traces (spectrometer data; see TraceHandler constructor)
            timeList : list of time arrays (see TraceHandler constructor)
            fieldList : list of field arrays (see TraceHandler constructor)
            stdevList : list of standard deviation arrays (see TraceHandler constructor)
            wvlList : list of wavelength arrays (spectrometer data; see TraceHandler constructor)
            spectrumList : list of spectrum arrays (spectrometer data; see TraceHandler constructor)
            traceHandlerList : list of TraceHandler objects

        """
        self.fsZeroPadding = 150
        self.traceHandlers = []
        if filenameList is not None:
            for i in range(len(filenameList)):
                if filenameSpectrumList is not None:
                    self.traceHandlers.append(
                        TraceHandler(
                            filename=filenameList[i],
                            filename_spectrum=filenameSpectrumList[i],
                        )
                    )
                else:
                    self.traceHandlers.append(TraceHandler(filename=filenameList[i]))
        elif timeList is not None and fieldList is not None:
            if stdevList is None:
                stdevList = [None] * len(timeList)
            if wvlList is None:
                wvlList = [None] * len(timeList)
            if spectrumList is None:
                spectrumList = [None] * len(timeList)
            check_equal_length(timeList, fieldList, stdevList, wvlList, spectrumList)
            for i in range(len(timeList)):
                self.traceHandlers.append(
                    TraceHandler(
                        time=timeList[i],
                        field=fieldList[i],
                        stdev=stdevList[i],
                        wvl=wvlList[i],
                        spectrum=spectrumList[i],
                    )
                )
        elif traceHandlerList is not None:
            for i in range(len(traceHandlerList)):
                if isinstance(traceHandlerList[i], TraceHandler):
                    self.traceHandlers.append(traceHandlerList[i])
                else:
                    raise ValueError(
                        "in function MultiTraceHandler.__init__() traceHandlerList is not a list of TraceHandler objects\n"
                    )
        else:
            raise ValueError(
                "in function MultiTraceHandler.__init__() too many arguments were None\n"
                "probably the constructor was erroneously used with no or too few arguments\n"
                "use either filenameList and filenameSpectrumList or timeList, fieldList or traceHandlerList\n"
            )

    def append_trace(
        self,
        filename=None,
        filename_spectrum=None,
        timeV=None,
        fieldV=None,
        stdevV=None,
        wvl=None,
        spectrum=None,
        traceHandler=None,
    ):
        """Append a new trace to the list. Usual rules apply.

        Args:
            filename: file path
            filename_spectrum: spectrum path
            timeV: time vector
            fieldV: field vector
            stdevV: standard deviation
            wvl: wavelength
            spectrum: spectrum
            traceHandler: TraceHandler instance

        """
        if filename_spectrum is not None and filename is not None:
            self.traceHandlers.append(TraceHandler(filename, filename_spectrum))
        elif filename is not None:
            self.traceHandlers.append(TraceHandler(filename))
        elif timeV is not None and fieldV is not None:
            self.traceHandlers.append(
                TraceHandler(
                    time=timeV, field=fieldV, stdev=stdevV, wvl=wvl, spectrum=spectrum
                )
            )
        elif traceHandler is not None:
            if isinstance(traceHandler, TraceHandler):
                self.traceHandlers.append(traceHandler)
            else:
                raise ValueError(
                    "in function MultiTraceHandler.append_trace() traceHandler is not a TraceHandler object\n"
                )
        else:
            raise ValueError(
                "in function MultiTraceHandler.append_trace() filename and filename_spectrum cannot be None\n"
                "it might be that the function was erroneously used with no arguments\n"
            )

    def flip_trace(self, index: int):
        """Flips the trace number 'index'.

        Args:
            index: int

        """
        if index >= len(self.traceHandlers):
            raise ValueError(
                "in function MultiTraceHandler.flip() index is out of range\n"
            )
        self.traceHandlers[index].fieldV = -self.traceHandlers[index].fieldV
        self.traceHandlers[index].fftFieldV = -self.traceHandlers[index].fftFieldV
        self.traceHandlers[index].complexFieldV = -self.traceHandlers[
            index
        ].complexFieldV

    def tukey_bandpass(
        self, lowWavelengthEdge, upWavelengthEdge, lowEdgeWidth, highEdgeWidth
    ):
        """Applies a bandpass filter the traces in the frequency domain using a tukey window. See traceHandler's docs."""
        for i in range(len(self.traceHandlers)):
            self.traceHandlers[i].fft_tukey_bandpass(
                lowWavelengthEdge, upWavelengthEdge, lowEdgeWidth, highEdgeWidth
            )

    def apply_spectrum(
        self, wvl=None, spectrum=None, CEP_shift=0.0, stripZeroPadding=True
    ):
        """Applies spectrum to the phase of the pulse. See traceHandler's docs."""
        for i in range(len(self.traceHandlers)):
            self.traceHandlers[i].apply_spectrum(
                wvl, spectrum, CEP_shift, stripZeroPadding
            )

    def plot_traces(
        self,
        low_lim=None,
        up_lim=None,
        labels=None,
        delay_shift=None,
        offset: float = 2.0,
        errorbar: bool = False,
        normalize: bool = True,
    ):
        """Plots all traces.

        Args:
            low_lim (float): for xaxis
            up_lim (float): for xaxis
            labels (list): list of labels for the legend
            delay_shift (list): list of delay offsets (one per trace)
            offset (float): y-axis offset between traces
            errorbar (bool): whether to plot errors. Default False
            normalize (bool): Default True

        """
        fig, ax = plt.subplots()
        if delay_shift is None:
            delay_shift = [0.0] * len(self.traceHandlers)
        check_equal_length(delay_shift, self.traceHandlers)

        for i in range(len(self.traceHandlers)):
            if normalize:
                norm_plot = self.traceHandlers[i].normalization_trace
            else:
                norm_plot = 1
            t, field = self.traceHandlers[i].get_trace()
            stdev_field = self.traceHandlers[i].get_stdev()
            if errorbar and stdev_field is not None:
                last_fill = ax.fill_between(
                    t - delay_shift[i],
                    offset * i + (field - stdev_field) / norm_plot,
                    offset * i + (field + stdev_field) / norm_plot,
                    label="_nolegend_",
                    alpha=0.3,
                )
                ax.plot(
                    t - delay_shift[i],
                    offset * i + field / norm_plot,
                    label="Trace " + str(i),
                    color=last_fill.get_facecolor(),
                    alpha=1.0,
                )
            else:
                ax.plot(
                    t - delay_shift[i],
                    offset * i + field / norm_plot,
                    label="Trace " + str(i),
                )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Field (Arb. unit)")

        if labels is not None:
            handles, labels_dump = ax.get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], loc="upper left")
        else:
            pass
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim, up_lim)
        return fig

    def plot_spectra(
        self, low_lim=50, up_lim=1000, labels=None, offset=0.015, logscale: bool = False
    ):
        """Plot all spectra.
        low_lim: lower limit wavelength (nm). Default: 50 nm
        up_lim: upper limit wavelength (nm). Default: 1000 nm
        labels: label list for the plot legend. Labels should be in the same order as the stored traceHandler objects
        offset: artificial offset between two spectra for display purposes. Default: 0.015
        logscale: (bool); whether to plot in a logscale.

        """
        fig, ax = plt.subplots()
        for i in range(len(self.traceHandlers)):
            self.traceHandlers[i].normalize_fft_spectrum()
            wvl, spctr = self.traceHandlers[i].get_spectrum_trace()
            if logscale:
                ax.plot(wvl, (offset**i) * spctr)
                ylow = np.mean(spctr[(wvl < up_lim) & (wvl > low_lim)]) / (offset**3)
                yup = np.mean(spctr[(wvl < up_lim) & (wvl > low_lim)]) * (
                    offset ** (len(self.traceHandlers))
                )
                ax.set_ylim(ylow, yup)
            else:
                ax.plot(wvl, i * offset + spctr)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        if labels is not None:
            handles, labels_dump = ax.get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], loc="upper right")
        else:
            pass
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim, up_lim)
        return fig
