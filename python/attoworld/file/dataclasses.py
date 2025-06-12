from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..numeric import interpolate, fwhm
from scipy import constants
import scipy.signal as sig
import copy

@dataclass
class Waveform:
    wave: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    dt: Optional[float] = None
    is_uniformly_spaced: bool = False

    def clone(self):
        return copy.deepcopy(self)
        
    def to_uniformly_spaced(self):
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

    def to_complex_spectrum(self, padding_factor: int = 1):
        spec = ComplexSpectrum()
        if self.wave is not None:
            if self.is_uniformly_spaced and self.dt is not None:
                spec.spectrum = np.fft.rfft(
                    self.wave,
                    n = self.wave.shape[0] * padding_factor,
                    axis = 0)
                spec.freq = np.fft.rfftfreq(self.wave.shape[0], d = self.dt/padding_factor)
            else:
                uniform_self = self.to_uniformly_spaced()
                if uniform_self.wave is not None and uniform_self.dt is not None:
                    spec.spectrum = np.fft.rfft(
                        uniform_self.wave,
                        n = uniform_self.wave.shape[0] * padding_factor,
                        axis = 0)
                    spec.freq = np.fft.rfftfreq(uniform_self.wave.shape[0], d = uniform_self.dt/padding_factor)
                else:
                    raise Exception("Interpolation failure.")

            spec.is_uniformly_spaced = True
            return spec
        else:
            raise Exception("No data to transform.")
    def to_intensity_spectrum(self, wavelength_scaled: bool = True):
        return self.to_complex_spectrum().to_intensity_spectrum(wavelength_scaled)
    
    def to_complex_envelope(self, f0: float = 0.0):
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.time is not None:
            analytic = sig.hilbert(uniform_self.wave)*np.exp(1j * 2*np.pi*f0 * uniform_self.time)
            return ComplexEnvelope(
                envelope = analytic,
                time = uniform_self.time,
                dt = uniform_self.dt,
                carrier_frequency = f0
            )
        else:
            raise Exception("Could not convert to complex envelope.")
        
    def get_envelope_fwhm(self):
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.dt is not None:
            return fwhm(np.abs(sig.hilbert(uniform_self.wave))**2, uniform_self.dt)
        else:
            raise Exception("No data to look at.")
            
    def get_field_squared_fwhm(self):
        uniform_self = self.to_uniformly_spaced()
        if uniform_self.wave is not None and uniform_self.dt is not None:
            return fwhm(uniform_self.wave**2, uniform_self.dt)
        else:
            raise Exception("No data to look at.")

@dataclass
class ComplexSpectrum:
    spectrum: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None
    is_uniformly_spaced: bool = True
    def clone(self):
        return copy.deepcopy(self)
    def to_intensity_spectrum(self, wavelength_scaled: bool = True):
        if self.spectrum is not None and self.freq is not None:
            output = IntensitySpectrum( 
                spectrum = np.abs(self.spectrum[self.freq>0.0]),
                phase = np.angle(self.spectrum[self.freq>0.0]),
                freq = self.freq[self.freq>0.0],
                wavelength = constants.speed_of_light/self.freq[self.freq>0.0],
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
    def clone(self):
        return copy.deepcopy(self)
        
@dataclass
class ComplexEnvelope:
    envelope: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    dt: Optional[float] = None
    carrier_frequency: float = 0.0
    def clone(self):
        return copy.deepcopy(self)
    def to_complex_spectrum(self, padding_factor: int = 1):
        if self.envelope is not None and self.dt is not None:
            return ComplexSpectrum(
                spectrum = np.fft.rfft(self.envelope, self.envelope.shape[0] * padding_factor),
                freq = np.fft.rfftfreq(self.envelope.shape[0] * padding_factor, self.dt) + self.carrier_frequency,
                is_uniformly_spaced = True
            )
