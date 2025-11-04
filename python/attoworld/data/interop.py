"""Operations for converting the data containters into each other."""

from typing import Optional

import numpy as np
import scipy.signal as sig
from scipy import constants

from ..numeric import (
    find_maximum_location,
)
from .complex_envelope import ComplexEnvelope
from .complex_spectrum import ComplexSpectrum
from .decorators import add_method
from .intensity_spectrum import IntensitySpectrum
from .waveform import Waveform


@add_method(Waveform, "to_complex_spectrum")
def waveform_to_complex_spectrum(self, padding_factor: int = 1):
    """Converts to a ComplexSpectrum class.

    Args:
        self: the class
        padding_factor (int): factor by which to expand the temporal length in the FFT, giving a smoother spectrum

    """
    if self.is_uniformly_spaced:
        new_spectrum = np.fft.rfft(
            self.wave, n=self.wave.shape[0] * padding_factor, axis=0
        )
        new_freq = np.fft.rfftfreq(self.wave.shape[0] * padding_factor, d=self.dt)
    else:
        uniform_self = self.to_uniformly_spaced()
        new_spectrum = np.fft.rfft(
            uniform_self.wave, n=uniform_self.wave.shape[0] * padding_factor, axis=0
        )
        new_freq = np.fft.rfftfreq(
            uniform_self.wave.shape[0] * padding_factor, d=uniform_self.dt
        )

    return ComplexSpectrum(spectrum=new_spectrum, freq=new_freq)


@add_method(Waveform, "to_intensity_spectrum")
def waveform_to_intensity_spectrum(
    self, wavelength_scaled: bool = True, padding_factor: int = 1
):
    """Converts to an intensity spectrum.

    Args:
        self: the class
        wavelength_scaled (bool): Correct the spectral intensities for plotting on a wavelength scale
        padding_factor (int): the factor by which the length will be multiplied

    """
    return self.to_complex_spectrum(padding_factor).to_intensity_spectrum(
        wavelength_scaled
    )


@add_method(Waveform, "to_time_derivative")
def waveform_to_time_derivative(self):
    """Return the time-derivative of the waveform."""
    return self.to_complex_spectrum().to_time_derivative().to_waveform()


@add_method(Waveform, "to_complex_envelope")
def waveform_to_complex_envelope(self, f0: float = 0.0):
    """Return a ComplexEnvelope class corresponding to the waveform.

    Args:
        self: the class
        f0 (float): central frequency to use when constructing the envelope. E.g. oscillation at this frequency will be cancelled.

    """
    uniform_self = self.to_uniformly_spaced()
    analytic = np.array(
        sig.hilbert(uniform_self.wave)
        * np.exp(-1j * 2 * np.pi * f0 * uniform_self.time)
    )
    return ComplexEnvelope(
        envelope=analytic,
        time=uniform_self.time,
        dt=uniform_self.dt,
        carrier_frequency=f0,
    )


@add_method(ComplexSpectrum, "to_waveform")
def complexspectrum_to_waveform(self):
    """Create a Waveform based on this complex spectrum."""
    wave = np.fft.irfft(self.spectrum, axis=0)
    dt = 0.5 / (self.freq[-1] - self.freq[0])
    time = dt * np.array(range(wave.shape[0]))
    return Waveform(wave=wave, time=time, dt=dt, is_uniformly_spaced=True)


@add_method(ComplexSpectrum, "to_centered_waveform")
def complexspectrum_to_centered_waveform(self):
    """Create a Waveform based on this complex spectrum and center it in the time window."""
    wave = np.fft.irfft(self.spectrum, axis=0)
    dt = 0.5 / (self.freq[-1] - self.freq[0])
    time = dt * np.array(range(wave.shape[0]))
    max_ind, max_val = find_maximum_location(np.abs(np.array(sig.hilbert(wave))))
    max_loc = dt * max_ind - 0.5 * time[-1]
    wave = np.fft.irfft(
        np.exp(-1j * self.freq * 2 * np.pi * max_loc) * self.spectrum, axis=0
    )
    return Waveform(wave=wave, time=time, dt=dt, is_uniformly_spaced=True)


@add_method(ComplexSpectrum, "to_intensity_spectrum")
def complexspectrum_to_intensity_spectrum(self, wavelength_scaled: bool = True):
    """Create an IntensitySpectrum based on the current ComplexSpectrum.

    Args:
        self: the class
        wavelength_scaled (bool): Apply the wavelength^-2 Jakobian such to correspond to W/nm spectrum

    """
    new_spectrum = np.array(np.abs(self.spectrum[self.freq > 0.0]) ** 2)
    new_freq = np.array(self.freq[self.freq > 0.0])
    new_wavelength = constants.speed_of_light / new_freq
    if wavelength_scaled:
        new_spectrum /= new_wavelength**2
    return IntensitySpectrum(
        spectrum=new_spectrum,
        phase=np.array(np.angle(self.spectrum[self.freq > 0.0])),
        freq=new_freq,
        wavelength=new_wavelength,
        is_frequency_scaled=wavelength_scaled,
    )


@add_method(IntensitySpectrum, "get_transform_limited_pulse")
def get_transform_limited_pulse(self, gate_level: Optional[float] = None):
    """Returns the transform-limited pulse corresponding to the spectrum.

    Args:
        self: the class
        gate_level (float): Apply a gate such that only values above gate_level*max(spectrum) are included

    Returns:
        ComplexEnvelope: the transform-limited pulse

    """
    if self.spectrum is not None:
        from ..spectrum import transform_limited_pulse_from_spectrometer

        spec = np.array(self.spectrum)
        lam = None
        if self.wavelength is None and self.freq is not None:
            lam = constants.speed_of_light / self.freq[self.freq > 0.0]
            spec = spec[self.freq > 0.0]
        elif self.wavelength is not None:
            lam = self.wavelength
        if lam is not None:
            if self.is_frequency_scaled:
                spec /= lam**2
            t, intensity = transform_limited_pulse_from_spectrometer(
                1e9 * lam, spec, gate_level
            )
            return ComplexEnvelope(envelope=np.sqrt(intensity), time=t, dt=t[1] - t[0])
    raise Exception("Missing data")


@add_method(ComplexEnvelope, "to_complex_spectrum")
def complex_envelope_to_complex_spectrum(
    self, padding_factor: int = 1
) -> ComplexSpectrum:
    """Returns a ComplexSpectrum based on the data."""
    return ComplexSpectrum(
        spectrum=np.fft.rfft(self.envelope, self.envelope.shape[0] * padding_factor),
        freq=np.fft.rfftfreq(self.envelope.shape[0] * padding_factor, self.dt)
        + self.carrier_frequency,
    )


@add_method(ComplexEnvelope, "to_waveform")
def complex_envelope_to_waveform(
    self, interpolation_factor: int = 1, CEP_shift: float = 0.0
) -> Waveform:
    """Returns a Waveform based on the data."""
    output_dt = self.dt / interpolation_factor
    output_time = self.time[0] + output_dt * np.array(
        range(self.time.shape[0] * interpolation_factor)
    )
    if interpolation_factor != 1:
        output_envelope = np.array(
            sig.resample(np.real(self.envelope), output_time.shape[0])
        ) + 1j * np.array(sig.resample(np.imag(self.envelope), output_time.shape[0]))
    else:
        output_envelope = self.envelope
    return Waveform(
        wave=np.real(
            np.exp(
                1j * self.carrier_frequency * 2 * np.pi * output_time - 1j * CEP_shift
            )
            * output_envelope
        ),
        time=output_time,
        dt=output_dt,
        is_uniformly_spaced=True,
    )
