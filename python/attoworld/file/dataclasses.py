from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..numeric import interpolate, fwhm, find_maximum_location
from ..spectrum import wavelength_to_frequency, frequency_to_wavelength, transform_limited_pulse_from_spectrometer
from scipy import constants
import scipy.signal as sig
import copy


@dataclass
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
            window_desc: String or tuple describing the desired window in the same format as scipy.signals.windows.get_window.

        Examples:
            >>> waveform_with_tukey_window = waveform.to_windowed('tukey')
            >>> waveform_with_supergaussian_window = waveform.to_windowed(('general_gaussian', 2, 500))
        """
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None:
            uniform_self.wave *= sig.windows.get_window(window_desc,Nx=uniform_self.wave.shape[0])
        return uniform_self
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
        return self.copy().to_complex_spectrum().to_bandpassed(frequency, sigma, order).to_waveform()
    def to_complex_spectrum(self, padding_factor: int = 1):
        """
        Converts to a ComplexSpectrum class

        Args:
            padding_factor (int): factor by which to expand the temporal length in the FFT, giving a smoother spectrum
        """
        spec = ComplexSpectrum()
        if self.wave is not None:
            if self.is_uniformly_spaced and self.dt is not None:
                spec.spectrum = np.fft.rfft(
                    self.wave,
                    n = self.wave.shape[0] * padding_factor,
                    axis = 0)
                spec.freq = np.fft.rfftfreq(self.wave.shape[0] * padding_factor, d = self.dt)
            else:
                uniform_self = self.to_uniformly_spaced()
                if uniform_self.wave is not None and uniform_self.dt is not None:
                    spec.spectrum = np.fft.rfft(
                        uniform_self.wave,
                        n = uniform_self.wave.shape[0] * padding_factor,
                        axis = 0)
                    spec.freq = np.fft.rfftfreq(uniform_self.wave.shape[0] * padding_factor, d = uniform_self.dt)
                else:
                    raise Exception("Interpolation failure.")
            return spec
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
        return self.to_complex_spectrum().time_derivative().to_waveform()

    def to_normalized(self):
        if self.wave is not None:
            max_loc ,max_val = find_maximum_location(np.abs(sig.hilbert(self.wave)))
            return Waveform(wave=self.wave/max_val, time=self.time, dt=self.dt, is_uniformly_spaced = self.is_uniformly_spaced)
        else:
            raise Exception("No data")

    def to_complex_envelope(self, f0: float = 0.0):
        """
        Convert to a ComplexEnvelope class

        Args:
            f0 (float): central frequency to use when constructing the envelope. E.g. oscillation at this frequency will be cancelled.
        """
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.time is not None:
            analytic = sig.hilbert(uniform_self.wave)*np.exp(-1j * 2*np.pi*f0 * uniform_self.time)
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

@dataclass
class ComplexSpectrum:
    spectrum: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None

    def copy(self):
        return copy.deepcopy(self)
    def time_derivative(self):
        if self.spectrum is not None and self.freq is not None:
            d_dt = 1j * 2 * np.pi * self.freq * self.spectrum
            return ComplexSpectrum(spectrum=d_dt, freq=np.array(self.freq))
        else:
            raise Exception("No data.")

    def to_bandpassed(self, frequency: float, sigma: float, order:int = 4):
        r"""
        Apply a supergaussian bandpass filter to the spectrum, of the form
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian
        """
        new_self = self.copy()
        if new_self.spectrum is not None and new_self.freq is not None:
            new_self.spectrum *= np.exp(-(new_self.freq - frequency)**order/(2*sigma**order))
        return new_self
    def to_waveform(self):
        if self.spectrum is not None and self.freq is not None:
            wave = np.fft.irfft(self.spectrum, axis=0)
            dt = 0.5/(self.freq[-1]-self.freq[0])
            time = dt * np.array(range(wave.shape[0]))
            return Waveform(wave=wave,time=time,dt=dt, is_uniformly_spaced=True)
        else:
            raise Exception("No data to transform")

    def to_intensity_spectrum(self, wavelength_scaled: bool = True):
        if self.spectrum is not None and self.freq is not None:
            output = IntensitySpectrum(
                spectrum = np.array(np.abs(self.spectrum[self.freq>0.0])**2),
                phase = np.array(np.angle(self.spectrum[self.freq>0.0])),
                freq = np.array(self.freq[self.freq>0.0]),
                wavelength = constants.speed_of_light/np.array(self.freq[self.freq>0.0]),
                is_frequency_scaled = wavelength_scaled)
            if wavelength_scaled and output.wavelength is not None:
                output.spectrum /= output.wavelength**2
        else:
            raise Exception("Insufficient data to make intensity spectrum.")
        return output

@dataclass
class IntensitySpectrum:
    spectrum: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None
    wavelength: Optional[np.ndarray] = None
    is_frequency_scaled: bool = False
    def copy(self):
        return copy.deepcopy(self)
    def wavelength_nm(self):
        if self.wavelength is not None:
            return 1e9 * self.wavelength
        else:
           return None
    def wavelength_micron(self):
        if self.wavelength is not None:
            return 1e6 * self.wavelength
        else:
           return None
    def from_spectrometer_spectrum_nanometers(self, wavelengths_nanometers: np.ndarray, spectrum: np.ndarray):
        self.spectrum = spectrum
        self.wavelength = 1e-9 * wavelengths_nanometers
        self.is_frequency_scaled = False
        self.phase = None
        self.freq = None
        self.phase = None
    def get_transform_limited_pulse(self, gate_level: Optional[float] = None):
        if self.spectrum is not None:
            spectrum = self.spectrum
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
        normalized_self = self.copy()
        if normalized_self.spectrum is not None:
            normalized_self.spectrum /= np.max(normalized_self.spectrum)
        return normalized_self
@dataclass
class ComplexEnvelope:
    envelope: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    dt: Optional[float] = None
    carrier_frequency: float = 0.0
    def copy(self):
        return copy.deepcopy(self)
    def get_fwhm(self) -> float:
        if self.envelope is not None and self.dt is not None:
            return fwhm(np.abs(self.envelope)**2, self.dt)
        else:
            raise Exception("Tried to take FWHM of data that doesn't exist.")

    def to_complex_spectrum(self, padding_factor: int = 1):
        if self.envelope is not None and self.dt is not None:
            return ComplexSpectrum(
                spectrum = np.fft.rfft(self.envelope, self.envelope.shape[0] * padding_factor),
                freq = np.fft.rfftfreq(self.envelope.shape[0] * padding_factor, self.dt) + self.carrier_frequency
            )
        else:
            raise Exception("Tried to convert non-existent data.")
    def to_waveform(self, interpolation_factor: int = 1, CEP_shift: float = 0.0):
        if self.envelope is not None and self.dt is not None and self.time is not None:
            output_dt = self.dt / interpolation_factor
            output_time = self.time[0] + output_dt * np.array(range(self.time.shape[0] * interpolation_factor))
            if interpolation_factor != 1:
                output_envelope = sig.resample(np.real(self.envelope), output_time.shape[0]) + 1j * sig.resample(np.imag(self.envelope), output_time.shape[0])
            else:
                output_envelope = self.envelope
            return Waveform(
                wave = np.real(np.exp(1j * self.carrier_frequency * 2 * np.pi * output_time - 1j * CEP_shift) * output_envelope),
                time = output_time,
                dt = output_dt,
                is_uniformly_spaced = True
            )
