from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..numeric import interpolate, fwhm, find_maximum_location
from ..spectrum import wavelength_to_frequency, frequency_to_wavelength, transform_limited_pulse_from_spectrometer
from scipy import constants
import scipy.signal as sig
import copy

def copy_if_not_none(data):
    """
    Helper function to handle optionals that should be deep-copied
    """
    return copy.deepcopy(data) if data is not None else None

@dataclass(frozen=True, slots=True)
class Spectrogram:
    data: np.ndarray
    time: np.ndarray
    freq: np.ndarray

@dataclass(frozen=True, slots=True)
class Waveform:
    """
    Contains data describing an electric field waveform.
    In SI units.

    Attributes:
        wave (np.ndarray): the waveform data
        time (np.ndarray): time vector corresponding to the wave array
        dt (float): the time step of time, if it is uniformly spaced
        is_uniformly_spaced (bool): says whether or not the time steps are uniform
    """
    wave: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    dt: Optional[float] = None
    is_uniformly_spaced: bool = False

    def copy(self):
        """
        Returns a deep copy
        """
        return copy.deepcopy(self)

    def time_fs(self):
        if self.time is not None:
            return 1e15 * self.time
        else:
            raise Exception("No data")

    def to_uniformly_spaced(self):
        """
        Returns a version of the struct with uniformly spaced time
        """
        if self.is_uniformly_spaced:
            return self
        else:
            if self.wave is not None and self.time is not None:
                timesteps = np.abs(np.diff(self.time))
                new_dt = np.min(timesteps[timesteps>0.0])
                new_time_length = self.time[-1]-self.time[0]
                new_time = self.time[0] + new_dt * np.array(range(int(new_time_length/new_dt)))
                return Waveform(
                    wave = interpolate(new_time, self.time, self.wave),
                    time = new_time,
                    dt = new_dt,
                    is_uniformly_spaced = True)
            else:
                raise Exception("Uninitialized data.")

    def to_windowed(self, window_desc):
        """
        Create a windowed version of the waveform. Output will be uniformly spaced in time, even if current state isn't.

        Args:
            window_desc: String or tuple describing the desired window in the same format as scipy.signal.windows.get_window.

        Examples:
            >>> waveform_with_tukey_window = waveform.to_windowed('tukey')
            >>> waveform_with_supergaussian_window = waveform.to_windowed(('general_gaussian', 2, 500))
        """
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None:
            new_wave = uniform_self.wave * sig.windows.get_window(window_desc,Nx=uniform_self.wave.shape[0])
            return Waveform(wave=new_wave, time=uniform_self.time, dt=uniform_self.dt, is_uniformly_spaced=True)
        else:
            raise Exception("No data to window.")
    def to_bandpassed(self, frequency: float, sigma: float, order: int = 4):
        r"""
        Apply a bandpass filter, with the same spec as to_bandpassed method of ComplexSpectrum:
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian
        """
        return self.to_complex_spectrum().to_bandpassed(frequency, sigma, order).to_waveform()
    def to_complex_spectrum(self, padding_factor: int = 1):
        """
        Converts to a ComplexSpectrum class

        Args:
            padding_factor (int): factor by which to expand the temporal length in the FFT, giving a smoother spectrum
        """
        new_spectrum = None
        new_freq = None
        if self.wave is not None:
            if self.is_uniformly_spaced and self.dt is not None:
                new_spectrum = np.fft.rfft(
                    self.wave,
                    n = self.wave.shape[0] * padding_factor,
                    axis = 0)
                new_freq = np.fft.rfftfreq(self.wave.shape[0] * padding_factor, d = self.dt)
            else:
                uniform_self = self.to_uniformly_spaced()
                if uniform_self.wave is not None and uniform_self.dt is not None:
                    new_spectrum = np.fft.rfft(
                        uniform_self.wave,
                        n = uniform_self.wave.shape[0] * padding_factor,
                        axis = 0)
                    new_freq = np.fft.rfftfreq(uniform_self.wave.shape[0] * padding_factor, d = uniform_self.dt)
                else:
                    raise Exception("Interpolation failure.")
            return ComplexSpectrum(spectrum=new_spectrum, freq=new_freq)
        else:
            raise Exception("No data to transform.")
    def to_intensity_spectrum(self, wavelength_scaled: bool = True, padding_factor: int = 1):
        """
        Converts to an intensity spectrum

        Args:
            wavelength_scaled (bool): Correct the spectral intensities for plotting on a wavelength scale
        """
        return self.to_complex_spectrum(padding_factor).to_intensity_spectrum(wavelength_scaled)
    def to_time_derivative(self):
        """
        Return the time-derivative of the waveform
        """
        return self.to_complex_spectrum().to_time_derivative().to_waveform()

    def to_normalized(self):
        """
        Return a normalized version of the waveform
        """
        if self.wave is not None:
            max_loc ,max_val = find_maximum_location(np.abs(np.array(sig.hilbert(self.wave))))
            return Waveform(wave=self.wave/max_val,
                time=copy_if_not_none(self.time),
                dt=self.dt,
                is_uniformly_spaced = self.is_uniformly_spaced)
        else:
            raise Exception("No data")

    def to_complex_envelope(self, f0: float = 0.0):
        """
        Return a ComplexEnvelope class corresponding to the waveform

        Args:
            f0 (float): central frequency to use when constructing the envelope. E.g. oscillation at this frequency will be cancelled.
        """
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.time is not None:
            analytic = np.array(sig.hilbert(uniform_self.wave)*np.exp(-1j * 2*np.pi*f0 * uniform_self.time))
            return ComplexEnvelope(
                envelope = analytic,
                time = uniform_self.time,
                dt = uniform_self.dt,
                carrier_frequency = f0
            )
        else:
            raise Exception("Could not convert to complex envelope.")

    def get_envelope_fwhm(self) -> float:
        """
        Get the full-width-at-half-maximum of the intensity envelope.

        Returns:
            float: the FWHM
        """
        return self.to_complex_envelope().get_fwhm()

    def get_field_squared_fwhm(self):
        """
        Get the full-width-at-half-maximum of the square of the field

        Returns:
            float: the FWHM
        """
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.dt is not None:
            return fwhm(uniform_self.wave**2, uniform_self.dt)
        else:
            raise Exception("No data to look at.")

@dataclass(frozen=True, slots=True)
class ComplexSpectrum:
    spectrum: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None

    def copy(self):
        return copy.deepcopy(self)
    def to_time_derivative(self):
        r"""Return a ComplexSpectrum corresponding to the time derivative (multiply by $i\omega$)"""
        if self.spectrum is not None and self.freq is not None:
            d_dt = 1j * 2 * np.pi * self.freq * self.spectrum
            return ComplexSpectrum(spectrum=d_dt, freq=np.array(self.freq))
        else:
            raise Exception("No data.")

    def to_bandpassed(self, frequency: float, sigma: float, order:int = 4):
        r"""
        Return the complex spectrum after applying a supergaussian bandpass filter to the spectrum, of the form
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian
        """
        if self.spectrum is not None and self.freq is not None:
            new_spectrum = self.spectrum * np.exp(-(self.freq - frequency)**order/(2*sigma**order))
            return ComplexSpectrum(spectrum=new_spectrum, freq = np.array(self.freq))
        else:
            raise Exception("No data to bandpass.")
    def to_waveform(self):
        """
        Create a Waveform based on this complex spectrum.
        """
        if self.spectrum is not None and self.freq is not None:
            wave = np.fft.irfft(self.spectrum, axis=0)
            dt = 0.5/(self.freq[-1]-self.freq[0])
            time = dt * np.array(range(wave.shape[0]))
            return Waveform(wave=wave,time=time,dt=dt, is_uniformly_spaced=True)
        else:
            raise Exception("No data to transform")

    def to_centered_waveform(self):
        if self.spectrum is not None and self.freq is not None:
            wave = np.fft.irfft(self.spectrum, axis=0)
            dt = 0.5/(self.freq[-1]-self.freq[0])
            time = dt * np.array(range(wave.shape[0]))
            max_ind, max_val = find_maximum_location(np.abs(np.array(sig.hilbert(wave))))
            max_loc = dt * max_ind - 0.5 * time[-1]
            wave = np.fft.irfft(np.exp(-1j * self.freq * 2*np.pi* max_loc)*self.spectrum, axis=0)
            return Waveform(wave=wave,time=time,dt=dt, is_uniformly_spaced=True)
        else:
            raise Exception("No data to transform")

    def to_intensity_spectrum(self, wavelength_scaled: bool = True):
        """Create an IntensitySpectrum based on the current ComplexSpectrum

        Args:
            wavelength_scaled (bool): Apply the wavelength^-2 Jakobian such to correspond to W/nm spectrum"""
        if self.spectrum is not None and self.freq is not None:
            new_spectrum = np.array(np.abs(self.spectrum[self.freq>0.0])**2)
            new_freq = np.array(self.freq[self.freq>0.0])
            new_wavelength = constants.speed_of_light/new_freq
            if wavelength_scaled:
                new_spectrum /= new_wavelength**2
            return IntensitySpectrum(
                spectrum = new_spectrum,
                phase = np.array(np.angle(self.spectrum[self.freq>0.0])),
                freq = new_freq,
                wavelength = new_wavelength,
                is_frequency_scaled = wavelength_scaled)
        else:
            raise Exception("Insufficient data to make intensity spectrum.")

@dataclass(frozen=True, slots=True)
class IntensitySpectrum:
    """
    Contains an intensity spectrum - real valued. SI units

    Attributes:
        spectrum (Optional[np.ndarray]): the spectral intensity
        phase (Optional[np.ndarray]): the spectral phase
        freq (Optional[np.ndarray]): the frequencies corresponding to the spectrum
        wavelength (Optional[np.ndarray]): the wavelengths
        is_frequency_scaled (bool): has the lamda^-2 Jakobian been applied to Fourier transformed data?
    """
    spectrum: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None
    wavelength: Optional[np.ndarray] = None
    is_frequency_scaled: bool = False
    def copy(self):
        return copy.deepcopy(self)
    def wavelength_nm(self):
        """
        Returns:
            Optional[np.ndarray]: the wavelengths in nm
        """
        if self.wavelength is not None:
            return 1e9 * self.wavelength
        else:
           return None
    def wavelength_micron(self):
        """
        Returns:
            Optional[np.ndarray]: the wavelengths in microns
        """
        if self.wavelength is not None:
            return 1e6 * self.wavelength
        else:
           return None
    def from_spectrometer_spectrum_nanometers(self, wavelengths_nanometers: np.ndarray, spectrum: np.ndarray):
        """
        Generate an instance based on a spepctrum array and wavelength array in nm (typical spectrometer data)

        Args:
            wavelengths_nanometers: the wavelengths in nanometers
            spectrum: the spectral intensity
        """
        return IntensitySpectrum(spectrum = spectrum, wavelength=1e-9 *wavelengths_nanometers)
    def get_transform_limited_pulse(self, gate_level: Optional[float] = None):
        """
        Returns the transform-limited pulse corresponding to the spectrum.

        Args:
            gate_level (float): Apply a gate such that only values above gate_level*max(spectrum) are included

        Returns:
            ComplexEnvelope: the transform-limited pulse
        """
        if self.spectrum is not None:
            spectrum = np.array(self.spectrum)
            lam = None
            if self.wavelength is None and self.freq is not None:
                lam = constants.speed_of_light / self.freq[self.freq > 0.0]
                spectrum = spectrum[self.freq>0.0]
            elif self.wavelength is not None:
                lam = self.wavelength
            if lam is not None:
                if self.is_frequency_scaled:
                    spectrum /= lam**2
                t, intensity = transform_limited_pulse_from_spectrometer(1e9*lam, spectrum, gate_level)
                return ComplexEnvelope(
                    envelope = np.sqrt(intensity),
                    time = t,
                    dt = t[1]-t[0]
                )
        raise Exception("Missing data")
    def get_wavelength_spectrum(self):
        """
        Return a wavelength-scaled spectrum, independent of the state of the current instance

        Returns:
            np.ndarray, np.ndarray: wavelength and spectrum
        """
        if self.is_frequency_scaled:
            if self.freq is not None and self.spectrum is not None:
                return frequency_to_wavelength(self.freq, self.spectrum)
            else:
                raise Exception("Missing data")
        else:
            if self.wavelength is not None and self.spectrum is not None:
                return self.wavelength, self.spectrum
            else:
                raise Exception("Missing data")

    def get_frequency_spectrum(self):
        """
        Return a frequency-scaled spectrum, independent of the state of the current instance

        Returns:
            np.ndarray, np.ndarray: wavelength and spectrum
        """
        if self.is_frequency_scaled:
            if self.freq is not None and self.spectrum is not None:
                return self.freq, self.spectrum
            else:
                raise Exception("Missing data")
        else:
            if self.wavelength is not None and self.spectrum is not None:
                return wavelength_to_frequency(1e9*self.wavelength, self.spectrum)
            else:
                raise Exception("Missing data")
    def to_normalized(self):
        """
        Returns a normalized version of the current instance.
        """
        if self.spectrum is not None:
            normalized_spectrum = self.spectrum / np.max(self.spectrum)
        else:
            normalized_spectrum = None
        return IntensitySpectrum(spectrum = normalized_spectrum,
            phase = copy_if_not_none(self.phase),
            freq = copy_if_not_none(self.freq),
            wavelength = copy_if_not_none(self.wavelength),
            is_frequency_scaled = self.is_frequency_scaled)

@dataclass(frozen=True, slots=True)
class ComplexEnvelope:
    """
    Data corresponding to a complex envelope of a pulse, e.g. from a FROG measurement.

    Attributes:
        envelope (Optional[np.ndarray]): the complex envelope
        time: (Optional[np.ndarray]): the time array
        dt (float): the time step
        carrier_frequency (float): the carrier frequency of the envelope
    """

    envelope: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    dt: Optional[float] = None
    carrier_frequency: float = 0.0
    def copy(self):
        return copy.deepcopy(self)
    def get_fwhm(self) -> float:
        """
        Full-width-at-half-maximum value of the envelope
        Returns:
            float: the fwhm
        """
        if self.envelope is not None and self.dt is not None:
            return fwhm(np.abs(self.envelope)**2, self.dt)
        else:
            raise Exception("Tried to take FWHM of data that doesn't exist.")

    def to_complex_spectrum(self, padding_factor: int = 1) -> ComplexSpectrum:
        """
        Returns a ComplexSpectrum based on the data
        """
        if self.envelope is not None and self.dt is not None:
            return ComplexSpectrum(
                spectrum = np.fft.rfft(self.envelope, self.envelope.shape[0] * padding_factor),
                freq = np.fft.rfftfreq(self.envelope.shape[0] * padding_factor, self.dt) + self.carrier_frequency
            )
        else:
            raise Exception("Tried to convert non-existent data.")
    def to_waveform(self, interpolation_factor: int = 1, CEP_shift: float = 0.0) -> Waveform:
        """
        Returns a Waveform based on the data
        """
        if self.envelope is not None and self.dt is not None and self.time is not None:
            output_dt = self.dt / interpolation_factor
            output_time = self.time[0] + output_dt * np.array(range(self.time.shape[0] * interpolation_factor))
            if interpolation_factor != 1:
                output_envelope = sig.resample(np.real(self.envelope), output_time.shape[0]) + 1j * np.array(sig.resample(np.imag(self.envelope), output_time.shape[0]))
            else:
                output_envelope = self.envelope
            return Waveform(
                wave = np.real(np.exp(1j * self.carrier_frequency * 2 * np.pi * output_time - 1j * CEP_shift) * output_envelope),
                time = output_time,
                dt = output_dt,
                is_uniformly_spaced = True
            )
        else:
            raise Exception("Not enough data to make a Waveform")
