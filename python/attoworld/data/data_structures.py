from dataclasses import dataclass, is_dataclass
import numpy as np
from typing import Optional, Tuple
from ..numeric import (
    interpolate,
    fwhm,
    find_maximum_location,
    derivative,
    block_binning_2d,
    block_binning_1d,
)
from ..plot import label_letter
from scipy import constants
import scipy.signal as sig
import scipy.optimize
import copy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import yaml
from .. import spectrum

def yaml_io(cls):
    """
    Adds functions to save and load the dataclass as yaml
    """

    def from_dict(cls, data: dict):
        """
        Takes a dict and makes an instance of the class

        Args:
            data (dict): the result of a call of .to_dict on the class
        """

        def handle_complex_array(serialized_array) -> np.ndarray:
            """Helper function to deserialize numpy arrays, handling complex types"""
            if isinstance(serialized_array, list) and all(
                isinstance(item, dict) and "re" in item and "im" in item
                for item in serialized_array
            ):
                return np.array(
                    [complex(item["re"], item["im"]) for item in serialized_array],
                    dtype=np.complex128,
                )
            return np.array(serialized_array)

        loaded_data = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_type is np.ndarray:
                loaded_data[field_name] = handle_complex_array(data[field_name])
            elif is_dataclass(field_type):
                loaded_data[field_name] = field_type.from_dict(data[field_name])
            else:
                loaded_data[field_name] = data[field_name]
        return cls(**loaded_data)

    def load_yaml(cls, filename: str):
        """
        load from a yaml file

        Args:
            filename (str): path to the file
        """
        with open(filename, "r") as file:
            data = yaml.load(file, yaml.SafeLoader)
            return cls.from_dict(data)

    def save_yaml(instance, filename: str):
        """
        save to a yaml file

        Args:
            filename (str): path to the file
        """
        data_dict = instance.to_dict()
        with open(filename, "w") as file:
            yaml.dump(data_dict, file)

    def to_dict(instance):
        """
        serialize the class into a dict
        """
        data_dict = {}
        for field_name, field_type in instance.__annotations__.items():
            field_value = getattr(instance, field_name)
            if field_type is np.ndarray:
                if field_value.dtype == np.complex128:
                    data_dict[field_name] = [
                        {"re": num.real, "im": num.imag} for num in field_value.tolist()
                    ]
                else:
                    data_dict[field_name] = field_value.tolist()
            elif is_dataclass(field_type):
                data_dict[field_name] = field_value.to_dict()
            elif field_type is np.float64 or field_type is float:
                data_dict[field_name] = float(field_value)
            else:
                data_dict[field_name] = field_value
        return data_dict

    cls.from_dict = classmethod(from_dict)
    cls.load_yaml = classmethod(load_yaml)
    cls.to_dict = to_dict
    cls.save_yaml = save_yaml
    return cls


@yaml_io
@dataclass(frozen=True, slots=True)
class SpectrometerCalibration:
    """
    Set of data describing a spectrometer calibration

    Attributes:
        intensity_factors (np.ndarray): the intensity correction factors (weights)
        corrected_wavelengths (np.ndarray): the corrected wavelength array of the spectrometer
        corrected_frequencies (np.ndarray): the corrected frequency array of the spectrometer (c/wavelengths)
    """

    intensity_factors: np.ndarray
    corrected_wavelengths: np.ndarray
    corrected_frequencies: np.ndarray

    def apply_to_spectrum(self, spectrum_in):
        """
        Applies itself to an intensity spectrum:

        Args:
            spectrum_in (IntensitySpectrum): the spectrum to be calibrated

        Returns:
            IntensitySpectrum: the calibrated spectrum
        """
        intensity_out = spectrum_in.spectrum * self.intensity_factors
        return IntensitySpectrum(
            spectrum=intensity_out,
            phase=spectrum_in.phase,
            wavelength=self.corrected_wavelengths,
            freq=self.corrected_frequencies,
        )

    def apply_to_spectrogram(self, spectrogram_in):
        """
        Applies itself to an intensity spectrum:

        Args:
            spectrum_in (Spectrogram): the spectrogram to be calibrated

        Returns:
            Spectrogram: the calibrated spectrogram
        """
        data_out = self.intensity_factors[:, np.newaxis] * spectrogram_in.data
        return Spectrogram(
            data=data_out, freq=self.corrected_frequencies, time=spectrogram_in.time
        )

    def save_npz(self, filepath):
        """
        Save to an npz file

        Args:
            filepath: path to save to
        """
        np.savez(
            filepath,
            intensity_factors=self.intensity_factors,
            corrected_wavelengths=self.corrected_wavelengths,
            corrected_frequencies=self.corrected_frequencies,
        )

    @staticmethod
    def from_npz(filepath):
        """
        Make an instance of the class from an npz file

        Args:
            filepath: path to the file

        Returns:
            SpectrometerCalibration: the calibration in the file"""
        npzfile = np.load(filepath)
        return SpectrometerCalibration(
            intensity_factors=npzfile["intensity_factors"],
            corrected_wavelengths=npzfile["corrected_wavelengths"],
            corrected_frequencies=npzfile["corrected_frequencies"],
        )

    @staticmethod
    def from_named(spectrometer: spectrum.CalibrationData):
        """
        Loads a calibration saved in the database

        Args:
            spectrometer (spectrum.CalibrationData): Value from the CalibrationData enum attoworld.spectrum.CalibrationData

        Returns:
            SpectrometerCalibration: the calibration associated with the enum value
        """
        return SpectrometerCalibration.from_npz(
            spectrum.get_calibration_path() / spectrometer.value
        )


@yaml_io
@dataclass(frozen=True, slots=True)
class Spectrogram:
    """
    Contains the data describing a spectrogram

    Attributes:
        data (np.ndarray): 2d spectrogram
        time (np.ndarray): time vector
        freq (np.ndarray): frequency vector"""

    data: np.ndarray
    time: np.ndarray
    freq: np.ndarray

    def lock(self):
        """
        Make the data immutable
        """
        self.data.setflags(write=False)
        self.time.setflags(write=False)
        self.freq.setflags(write=False)

    def save(self, filename):
        """
        Save in the .A.dat file format used by FROG .etc
        Args:
            filename: the file to be saved

        The file is structured like this:
        [number of wavelengths] [number of times]
        [minimum value of the trace] [maximum value of the trace]
        [array of wavelengths]
        [array of times]
        [data array as single column]
        """
        with open(filename, "w") as file:
            file.write(f"{len(self.freq)}\t{len(self.time)}\n")
            file.write(f"{np.min(self.data[:])}\t{np.max(self.data[:])}\n")
            for freq in self.freq:
                wavelength_nm = 1e9 * constants.speed_of_light / freq
                file.write(f"{wavelength_nm:15.15g}\n")
            for time in self.time:
                time_fs = 1e15 * time
                file.write(f"{time_fs:15.15g}\n")
            for x in self.data:
                for y in x:
                    file.write(f"{y:15.15g}\n")

    def to_block_binned(self, freq_bin: int, time_bin: int, method: str = "mean"):
        """
        Apply block-binning to the spectrogram.

        Args:
            freq_bin (int): block size for averaging in the frequency direction
            time_bin (int): block size for averaging in the time-direction
            method (str): can be ```mean``` or ```median```
        """
        return Spectrogram(
            data=block_binning_2d(self.data, time_bin, freq_bin, method),
            freq=block_binning_1d(self.freq, freq_bin, "mean"),
            time=block_binning_1d(self.time, time_bin, "mean"),
        )

    def to_per_frequency_dc_removed(self, extra_offset: float = 0.0):
        """Perform DC offset removal on a measured spectrogram, on a per-frequency basis

        Args:
            extra_offset (float): subtract a value from the entire array (negative values are always set to zero)

        Returns:
            Spectrogram: the spectrogram with offset removed."""
        new_data = np.array(self.data)
        new_data -= extra_offset
        new_data[new_data < 0.0] = 0.0
        for _i in range(new_data.shape[0]):
            new_data[_i, :] -= np.min(new_data[_i, :])

        return Spectrogram(data=new_data, time=self.time, freq=self.freq)

    def to_symmetrized(self):
        """
        Average the trace with a time-reversed copy. This might be useful for getting a reconstruction of difficult data, but keep in mind that the resulting measured trace will no longer represent the real measurement and should not be published as such.
        """
        return Spectrogram(
            data=0.5 * (self.data + np.fliplr(self.data)),
            time=self.time,
            freq=self.freq,
        )

    def to_removed_spatial_chirp(self):
        """
        Remove the effects of spatial chirp on an SHG-FROG trace by centering all single-frequency autocorrelations to the same time-zero
        """
        new_data = np.array(self.data)
        for i in range(len(self.freq)):
            total = np.sum(self.data[i, :])
            if total > 0.0:
                t0 = np.sum(self.time * self.data[i, :]) / total
                new_data[i, :] = interpolate(self.time + t0, self.time, self.data[i, :])

        return Spectrogram(data=new_data, time=self.time, freq=self.freq)

    def to_combined_and_binned(
        self,
        other,
        stitching_band: Tuple[float, float],
        dim: int = 64,
        dt: float = 5e-15,
        t0: Optional[Tuple[float, float]] = None,
        f0: float = 750e12,
    ):
        """
        Bin two different spectrograms, e.g. from different spectrometers, onto the time time/frequency grid

        Args:
            other: the other spectrogram
            stitching_band (Tuple[float, float]): the lower and upper frequency of the band where the two spectrometers should have equivalent response (hopefully there is one)
            dim (int): size of each size of the resulting square data
            dt (float): time step of the data
            t0: (Optional[Tuple[float, float]): time-zero of the data (this, and other). If not specified, will be calculated by the first moment of the time-distribution of the signal
            f0: (float): central frequency of the binned array
        """
        t0_self = None
        t0_other = None

        if t0 is not None:
            t0_self = t0[0]
            t0_other = t0[1]

        binned_self = self.to_binned(dim, dt, t0_self, f0)
        binned_other = other.to_binned(dim, dt, t0_other, f0)
        freq = binned_self.freq

        # add more logic here to combine the spectrograms
        stitching_band_integral_self = np.sum(
            binned_self.data[
                ((freq > stitching_band[0]) & (freq < stitching_band[1])), :
            ][:]
        )
        stitching_band_integral_other = np.sum(
            binned_other.data[
                ((freq > stitching_band[0]) & (freq < stitching_band[1])), :
            ][:]
        )
        weights_self = np.zeros(binned_self.freq.shape, dtype=float)
        weights_other = np.zeros(binned_other.freq.shape, dtype=float)
        other_multiplier = stitching_band_integral_self / stitching_band_integral_other
        for i in range(len(freq)):
            sum_self = np.sum(binned_self.data[i, :])
            sum_other = other_multiplier * np.sum(binned_other.data[i, :])
            total = sum_self + sum_other
            if total > 0.0:
                weight_self = sum_self / total
                weight_other = other_multiplier * sum_other / total
                weights_self[i] = weight_self
                weights_other[i] = weight_other

        return Spectrogram(
            data=weights_self[:, np.newaxis] * binned_self.data
            + weights_other[:, np.newaxis] * binned_other.data,
            time=binned_self.time,
            freq=binned_self.freq,
        )

    def to_binned(
        self,
        dim: int = 64,
        dt: float = 5e-15,
        t0: Optional[float] = None,
        f0: float = 750e12,
    ):
        """Bin a spectrogram to a FFT-appropriate shape

        Args:
            dim (int): size of each size of the resulting square data
            dt (float): time step of the data
            t0: (Optional[float]): time-zero of the data. If not specified, will be calculated by the first moment of the time-distribution of the signal
            f0: (float): central frequency of the binned array

        Returns:
            Spectrogram: the binned spectrogram
        """
        _t = np.array(range(dim)) * dt
        _t -= np.mean(_t)
        _f = np.fft.fftshift(np.fft.fftfreq(dim, d=dt) + f0)
        binned_data = np.zeros((dim, self.time.shape[0]), dtype=float)
        for _i in range(self.time.shape[0]):
            binned_data[:, _i] = interpolate(
                _f, self.freq, np.array(self.data[:, _i]), neighbors=2
            )
        binned_data /= np.max(binned_data[:])
        if t0 is None:
            ac = np.sum(binned_data, axis=0)
            t0 = np.sum(ac * self.time) / np.sum(ac)
        binned_data_square = np.zeros((dim, dim), dtype=float)
        for _i in range(dim):
            binned_data_square[_i, :] = interpolate(
                _t, self.time - t0, np.array(binned_data[_i, :]), neighbors=2
            )
        return Spectrogram(data=binned_data_square, time=_t, freq=_f)

    def plot(self, ax: Optional[Axes] = None):
        """
        Plot the spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        a = ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            self.data / np.max(self.data[:]),
            rasterized=True,
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Frequency (THz)")
        plt.colorbar(a)
        return fig

    def plot_log(self, ax: Optional[Axes] = None):
        """
        Plot the spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        logdata = np.array(self.data)
        logdata[self.data > 0.0] = np.log(self.data[self.data > 0.0])
        logdata[self.data < 0.0] = 0.0
        a = ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            logdata,
            rasterized=True,
            vmin=-11,
            vmax=0,
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Frequency (THz)")
        ax.grid(True, lw=1)
        plt.colorbar(a)
        return fig


@yaml_io
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

    wave: np.ndarray
    time: np.ndarray
    dt: float
    is_uniformly_spaced: bool = False

    def lock(self):
        """
        Make the data immutable
        """
        self.wave.setflags(write=False)
        self.time.setflags(write=False)

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
            timesteps = np.abs(np.diff(self.time))
            new_dt = np.min(timesteps[timesteps > 0.0])
            new_time_length = self.time[-1] - self.time[0]
            new_time = self.time[0] + new_dt * np.array(
                range(int(new_time_length / new_dt))
            )
            return Waveform(
                wave=interpolate(new_time, self.time, self.wave),
                time=new_time,
                dt=new_dt,
                is_uniformly_spaced=True,
            )

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
        new_wave = uniform_self.wave * sig.windows.get_window(
            window_desc, Nx=uniform_self.wave.shape[0]
        )
        return Waveform(
            wave=new_wave,
            time=uniform_self.time,
            dt=uniform_self.dt,
            is_uniformly_spaced=True,
        )

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
        return (
            self.to_complex_spectrum()
            .to_bandpassed(frequency, sigma, order)
            .to_waveform()
        )

    def to_complex_spectrum(self, padding_factor: int = 1):
        """
        Converts to a ComplexSpectrum class

        Args:
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

    def to_intensity_spectrum(
        self, wavelength_scaled: bool = True, padding_factor: int = 1
    ):
        """
        Converts to an intensity spectrum

        Args:
            wavelength_scaled (bool): Correct the spectral intensities for plotting on a wavelength scale
        """
        return self.to_complex_spectrum(padding_factor).to_intensity_spectrum(
            wavelength_scaled
        )

    def to_time_derivative(self):
        """
        Return the time-derivative of the waveform
        """
        return self.to_complex_spectrum().to_time_derivative().to_waveform()

    def to_normalized(self):
        """
        Return a normalized version of the waveform
        """
        max_loc, max_val = find_maximum_location(
            np.abs(np.array(sig.hilbert(self.wave)))
        )
        return Waveform(
            wave=self.wave / max_val,
            time=np.array(self.time),
            dt=self.dt,
            is_uniformly_spaced=self.is_uniformly_spaced,
        )

    def to_complex_envelope(self, f0: float = 0.0):
        """
        Return a ComplexEnvelope class corresponding to the waveform

        Args:
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
        return fwhm(uniform_self.wave**2, uniform_self.dt)


@yaml_io
@dataclass(frozen=True, slots=True)
class ComplexSpectrum:
    """
    Contains a complex spectrum, with spectral weights on a frequency scale

    Attributes:
        spectrum (np.ndarray): the spectrum in complex128 format
        freq (np.ndarray):
    """

    spectrum: np.ndarray
    freq: np.ndarray

    def lock(self):
        """
        Make the data immutable
        """
        self.spectrum.setflags(write=False)
        self.freq.setflags(write=False)

    def copy(self):
        return copy.deepcopy(self)

    def to_time_derivative(self):
        r"""Return a ComplexSpectrum corresponding to the time derivative (multiply by $i\omega$)"""
        d_dt = 1j * 2 * np.pi * self.freq * self.spectrum
        return ComplexSpectrum(spectrum=d_dt, freq=np.array(self.freq))

    def to_bandpassed(self, frequency: float, sigma: float, order: int = 4):
        r"""
        Return the complex spectrum after applying a supergaussian bandpass filter to the spectrum, of the form
        $e^{\frac{(f-f_0)^r}{2\sigma^r}}$
        where $f_0$ is the frequency argument, $\sigma$ is the sigma argument, and r is the order argument

        Args:
            frequency: the central frequency (Hz) of the bandpass
            sigma: the width of the bandpass (Hz)
            order: the order of the supergaussian
        """
        new_spectrum = self.spectrum * np.exp(
            -((self.freq - frequency) ** order) / (2 * sigma**order)
        )
        return ComplexSpectrum(spectrum=new_spectrum, freq=np.array(self.freq))

    def to_waveform(self):
        """
        Create a Waveform based on this complex spectrum.
        """
        wave = np.fft.irfft(self.spectrum, axis=0)
        dt = 0.5 / (self.freq[-1] - self.freq[0])
        time = dt * np.array(range(wave.shape[0]))
        return Waveform(wave=wave, time=time, dt=dt, is_uniformly_spaced=True)

    def to_centered_waveform(self):
        """
        Create a Waveform based on this complex spectrum and center it in the time window
        """
        wave = np.fft.irfft(self.spectrum, axis=0)
        dt = 0.5 / (self.freq[-1] - self.freq[0])
        time = dt * np.array(range(wave.shape[0]))
        max_ind, max_val = find_maximum_location(np.abs(np.array(sig.hilbert(wave))))
        max_loc = dt * max_ind - 0.5 * time[-1]
        wave = np.fft.irfft(
            np.exp(-1j * self.freq * 2 * np.pi * max_loc) * self.spectrum, axis=0
        )
        return Waveform(wave=wave, time=time, dt=dt, is_uniformly_spaced=True)

    def to_intensity_spectrum(self, wavelength_scaled: bool = True):
        """Create an IntensitySpectrum based on the current ComplexSpectrum

        Args:
            wavelength_scaled (bool): Apply the wavelength^-2 Jakobian such to correspond to W/nm spectrum"""
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


@yaml_io
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

    spectrum: np.ndarray
    phase: Optional[np.ndarray]
    freq: np.ndarray
    wavelength: np.ndarray
    is_frequency_scaled: bool = False

    def lock(self):
        """
        Make the data immutable
        """
        self.spectrum.setflags(write=False)
        self.freq.setflags(write=False)
        self.wavelength.setflags(write=False)
        if self.phase is not None:
            self.phase.setflags(write=False)

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

    @staticmethod
    def from_spectrometer_spectrum_nanometers(
        wavelengths_nanometers: np.ndarray, spectrum: np.ndarray
    ):
        """
        Generate an instance based on a spepctrum array and wavelength array in nm (typical spectrometer data)

        Args:
            wavelengths_nanometers: the wavelengths in nanometers
            spectrum: the spectral intensity
        """
        wavelengths = 1e-9 * wavelengths_nanometers
        freqs = constants.speed_of_light / wavelengths
        phase = np.zeros(freqs.shape, dtype=float)
        return IntensitySpectrum(
            spectrum=spectrum, wavelength=wavelengths, freq=freqs, phase=phase
        )

    def get_transform_limited_pulse(self, gate_level: Optional[float] = None):
        """
        Returns the transform-limited pulse corresponding to the spectrum.

        Args:
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
                return ComplexEnvelope(
                    envelope=np.sqrt(intensity), time=t, dt=t[1] - t[0]
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
                from ..spectrum import frequency_to_wavelength

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
                from ..spectrum import wavelength_to_frequency

                return wavelength_to_frequency(1e9 * self.wavelength, self.spectrum)
            else:
                raise Exception("Missing data")

    def to_normalized(self):
        """
        Returns a normalized version of the current instance.
        """

        normalized_spectrum = self.spectrum / np.max(self.spectrum)

        return IntensitySpectrum(
            spectrum=normalized_spectrum,
            phase=self.phase,
            freq=np.array(self.freq),
            wavelength=np.array(self.wavelength),
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def to_interpolated_wavelength(self, new_wavelengths: np.ndarray):
        new_spectrum = interpolate(new_wavelengths, self.wavelength, self.spectrum)
        if self.phase is not None:
            new_phase = interpolate(new_wavelengths, self.wavelength, self.phase)
        else:
            new_phase = None

        return IntensitySpectrum(
            spectrum=new_spectrum,
            wavelength=new_wavelengths,
            freq=constants.speed_of_light / new_wavelengths,
            phase=new_phase,
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def to_corrected_wavelength(self, new_wavelengths: np.ndarray):
        assert len(new_wavelengths) == len(self.wavelength)
        return IntensitySpectrum(
            spectrum=self.spectrum,
            wavelength=new_wavelengths,
            freq=constants.speed_of_light / new_wavelengths,
            phase=self.phase,
            is_frequency_scaled=self.is_frequency_scaled,
        )

    def plot_with_group_delay(
        self,
        ax: Optional[Axes] = None,
        phase_blanking: float = 0.05,
        shift_from_centered=True,
        xlim=None,
    ):
        """
        Plot the spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        start_index = np.argmax(self.spectrum > 0)
        intensity = self.spectrum[start_index::]
        freq = self.freq[start_index::]
        wl = constants.speed_of_light / freq
        if self.phase is not None:
            if shift_from_centered:
                shift = -0.5 / (freq[1] - freq[0])
                phase_shift = np.angle(np.exp(1j * 2 * np.pi * freq * shift))
                phase = np.unwrap(phase_shift + self.phase[start_index::])
            else:
                phase = np.unwrap(self.phase[start_index::])

        else:
            phase = np.zeros(intensity.shape, dtype=float)

        intensity /= np.max(intensity)
        intensity_line = ax.plot(1e9 * wl, intensity, label="Intensity")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        ax_phase = plt.twinx(ax)
        group_delay = (1e15 / (2 * np.pi)) * derivative(phase, 1) / (freq[1] - freq[2])
        assert isinstance(ax_phase, Axes)
        ax_phase.plot([], [])
        phase_line = ax_phase.plot(
            1e9 * wl[intensity > phase_blanking],
            group_delay[intensity > phase_blanking],
            "--",
            label="Group delay",
        )
        ax_phase.set_ylabel("Group delay (fs)")
        if xlim is not None:
            ax.set_xlim(xlim)
            ax_phase.set_xlim(xlim)
        lines = lines = intensity_line + phase_line
        ax.legend(lines, [str(line.get_label()) for line in lines])
        return fig


@yaml_io
@dataclass(frozen=True, slots=True)
class ComplexEnvelope:
    """
    Data corresponding to a complex envelope of a pulse, e.g. from a FROG measurement.

    Attributes:
        envelope (np.ndarray): the complex envelope
        time: (np.ndarray): the time array
        dt (float): the time step
        carrier_frequency (float): the carrier frequency of the envelope
    """

    envelope: np.ndarray
    time: np.ndarray
    dt: float
    carrier_frequency: float = 0.0

    def lock(self):
        """
        Make the data immutable
        """
        self.envelope.setflags(write=False)
        self.time.setflags(write=False)

    def time_fs(self):
        return 1e15 * self.time

    def copy(self):
        return copy.deepcopy(self)

    def get_fwhm(self) -> float:
        """
        Full-width-at-half-maximum value of the envelope
        Returns:
            float: the fwhm
        """
        return fwhm(np.abs(self.envelope) ** 2, self.dt)

    def to_complex_spectrum(self, padding_factor: int = 1) -> ComplexSpectrum:
        """
        Returns a ComplexSpectrum based on the data
        """

        return ComplexSpectrum(
            spectrum=np.fft.rfft(
                self.envelope, self.envelope.shape[0] * padding_factor
            ),
            freq=np.fft.rfftfreq(self.envelope.shape[0] * padding_factor, self.dt)
            + self.carrier_frequency,
        )

    def to_waveform(
        self, interpolation_factor: int = 1, CEP_shift: float = 0.0
    ) -> Waveform:
        """
        Returns a Waveform based on the data
        """
        output_dt = self.dt / interpolation_factor
        output_time = self.time[0] + output_dt * np.array(
            range(self.time.shape[0] * interpolation_factor)
        )
        if interpolation_factor != 1:
            output_envelope = sig.resample(
                np.real(self.envelope), output_time.shape[0]
            ) + 1j * np.array(
                sig.resample(np.imag(self.envelope), output_time.shape[0])
            )
        else:
            output_envelope = self.envelope
        return Waveform(
            wave=np.real(
                np.exp(
                    1j * self.carrier_frequency * 2 * np.pi * output_time
                    - 1j * CEP_shift
                )
                * output_envelope
            ),
            time=output_time,
            dt=output_dt,
            is_uniformly_spaced=True,
        )

    def plot(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """
        Plot the pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        time_ax = self.time_fs() - np.mean(self.time_fs())
        intensity = np.abs(self.envelope) ** 2
        intensity /= np.max(intensity)
        intensity_line = ax.plot(
            time_ax,
            intensity,
            label=f"Intensity, fwhm {1e15 * self.get_fwhm():0.1f} fs",
        )
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Intensity (Arb. unit)")
        inst_freq = (
            (1e-12 / (2 * np.pi))
            * derivative(np.unwrap(np.angle(self.envelope)), 1)
            / self.dt
        )
        ax_phase = plt.twinx(ax)
        assert isinstance(ax_phase, Axes)
        ax_phase.plot([], [])
        phase_line = ax_phase.plot(
            time_ax[intensity > phase_blanking],
            inst_freq[intensity > phase_blanking],
            "--",
            label="Inst. frequency",
        )
        ax_phase.set_ylabel("Inst. frequency (THz)")
        if xlim is not None:
            ax.set_xlim(xlim)
            ax_phase.set_xlim(xlim)
        lines = lines = intensity_line + phase_line
        ax.legend(lines, [str(line.get_label()) for line in lines])
        return fig


@yaml_io
@dataclass(frozen=True, slots=True)
class FrogData:
    """
    Stores data from a FROG measurement

    Attributes:
        spectrum (ComplexSpectrum): the reconstructed complex spectrum
        pulse (Waveform): time-domain reconstructed field
        measured_spectrogram (Spectrogram): measured (binned) data
        reconstructed_spectrogram (Spectrogram): spectrogram resulting from reconstructed field
        f0: the central frequency of the spectrum
        dt: the time step
    """

    spectrum: ComplexSpectrum
    pulse: Waveform
    measured_spectrogram: Spectrogram
    reconstructed_spectrogram: Spectrogram
    raw_reconstruction: np.ndarray
    f0: float
    dt: float

    def lock(self):
        """
        Make the data immutable
        """
        self.raw_reconstruction.setflags(write=False)
        self.spectrum.lock()
        self.pulse.lock()
        self.measured_spectrogram.lock()
        self.reconstructed_spectrogram.lock()

    def save(self, base_filename):
        """
        Save in the Trebino FROG format

        Args:
            base_filename: base of the file path; 4 files will be made from it: .A.dat, .Arecon.dat, .Ek.dat, and .Speck.dat
        """
        self.measured_spectrogram.save(base_filename + ".A.dat")
        self.reconstructed_spectrogram.save(base_filename + ".Arecon.dat")

        t = 1e15 * self.dt * np.array(range(len(self.raw_reconstruction)))
        t -= np.mean(t)
        f = np.fft.fftshift(np.fft.fftfreq(len(t), d=self.dt)) + self.f0
        lam = constants.speed_of_light / f
        raw_spec = np.fft.fftshift(np.fft.fft(self.raw_reconstruction))

        with open(base_filename + ".Ek.dat", "w") as time_file:
            for _i in range(len(self.raw_reconstruction)):
                time_file.write(
                    f"{t[_i]:.15g}\t{np.abs(self.raw_reconstruction[_i]) ** 2:.15g}\t{np.angle(self.raw_reconstruction[_i]):.15g}\t{np.real(self.raw_reconstruction[_i]):.15g}\t{np.imag(self.raw_reconstruction[_i]):.15g}\n"
                )

        with open(base_filename + ".Speck.dat", "w") as spec_file:
            for _i in range(len(self.raw_reconstruction)):
                spec_file.write(
                    f"{lam[_i]:.15g}\t{np.abs(raw_spec[_i]) ** 2:.15g}\t{np.angle(raw_spec[_i]):.15g}\t{np.real(raw_spec[_i]):.15g}\t{np.imag(raw_spec[_i]):.15g}\n"
                )

    def plot_measured_spectrogram(self, ax: Optional[Axes] = None, log: bool = False):
        """
        Plot the measured spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if log:
            self.measured_spectrogram.plot_log(ax)
        else:
            self.measured_spectrogram.plot(ax)
        ax.set_title("Measured")
        return fig

    def plot_reconstructed_spectrogram(
        self, ax: Optional[Axes] = None, log: bool = False
    ):
        """
        Plot the reconstructed spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if log:
            self.reconstructed_spectrogram.plot_log(ax)
        else:
            self.reconstructed_spectrogram.plot(ax)
        ax.set_title(
            f"Retrieved (G': {self.get_error():0.2e}; G: {self.get_G_error():0.2e})"
        )
        return fig

    def plot_pulse(
        self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None
    ):
        """
        Plot the reconstructed pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        return self.pulse.to_complex_envelope().plot(ax, phase_blanking, xlim)

    def plot_spectrum(
        self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None
    ):
        """
        Plot the reconstructed spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        return self.spectrum.to_intensity_spectrum().plot_with_group_delay(
            ax, phase_blanking=phase_blanking, shift_from_centered=True, xlim=xlim
        )

    def plot_all(
        self,
        phase_blanking=0.05,
        time_xlims=None,
        wavelength_xlims=None,
        figsize=None,
        log: bool = False,
    ):
        """
        Produce a 4-panel plot of the FROG results, combining calls to plot_measured_spectrogram(),
        plot_reconstructed_spectrogram(), plot_pulse() and plot_spectrum() as subplots, with letter labels.

        Args:
            phase_blanking: relative intensity at which to show phase information
            time_xlim: x-axis limits to pass to the plot of the pulse
            wavelength_xlim: x-axis limits to pass to the plot of the spectrum
            figsize: custom figure size
        """
        if figsize is None:
            default_figsize = plt.rcParams["figure.figsize"]
            figsize = (default_figsize[0] * 2, default_figsize[1] * 2)
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        self.plot_measured_spectrogram(ax[0, 0], log=log)
        label_letter("a", ax[0, 0])
        self.plot_reconstructed_spectrogram(ax[1, 0], log=log)
        label_letter("b", ax[1, 0])
        self.plot_pulse(ax[0, 1], xlim=time_xlims, phase_blanking=phase_blanking)
        label_letter("c", ax[0, 1])
        self.plot_spectrum(
            ax[1, 1], xlim=wavelength_xlims, phase_blanking=phase_blanking
        )
        label_letter("d", ax[1, 1])
        return fig

    def get_error(self) -> float:
        """
        Get the G' error of the reconstruction
        """
        norm_measured = np.linalg.norm(self.measured_spectrogram.data)
        norm_retrieved = np.linalg.norm(self.reconstructed_spectrogram.data)
        return np.sqrt(
            np.sum(
                (
                    self.measured_spectrogram.data[:] / norm_measured
                    - self.reconstructed_spectrogram.data[:] / norm_retrieved
                )
                ** 2
            )
            / np.sum((self.measured_spectrogram.data[:] / norm_measured) ** 2)
        )

    def get_G_error(self) -> float:
        """
        Get the G (note: no apostrophe) error. This one doesn't mean much, but is useful
        for comparing reconstructions of the same spectrogram between different programs.
        """
        return np.sqrt(
            (1.0 / float(len(self.measured_spectrogram.data[:]) ** 2))
            * np.sum(
                (
                    self.measured_spectrogram.data[:]
                    / np.max(self.measured_spectrogram.data[:])
                    - self.reconstructed_spectrogram.data[:]
                    / np.max(self.reconstructed_spectrogram.data[:])
                )
                ** 2
            )
        )

    def get_fwhm(self) -> float:
        """
        Get the full-width-at-half-max value of the reconstructed pulse
        """
        return self.pulse.get_envelope_fwhm()


@yaml_io
@dataclass
class CalibrationInput:
    wavelength_center: float
    wavelength_offset: float
    wavelength_slope: float
    amplitude_center: float
    amplitude_multiplier: float
    amplitude_slope: float
    amplitude_width: float
    amplitude_order: float
    roi_lowest: float
    roi_highest: float
    noise_level: float

    def get_wavelength_array(self):
        return np.array(
            [self.wavelength_center, self.wavelength_offset, self.wavelength_slope]
        )

    def get_amplitude_array(self):
        return np.array(
            [
                self.amplitude_center,
                self.amplitude_multiplier,
                self.amplitude_slope,
                self.amplitude_width,
                self.amplitude_order,
            ]
        )

    def plot(
        self,
        measurement: IntensitySpectrum,
        reference: IntensitySpectrum,
        plot_xmin=None,
        plot_xmax=None,
    ):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].plot(
            measurement.wavelength_nm(), measurement.spectrum, label="Measurement"
        )
        ax[0].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[0].set_xlabel("Wavelength (nm)")
        ax[0].set_ylabel("Intensity (Arb. unit)")
        ax[0].legend()
        initial_guess_calibration = generate_calibration_from_coeffs(
            self.get_amplitude_array(),
            self.get_wavelength_array(),
            measurement.wavelength,
        )
        initial_guess_spectrum = initial_guess_calibration.apply_to_spectrum(
            measurement
        )
        ax[1].plot(
            initial_guess_spectrum.wavelength_nm(),
            initial_guess_spectrum.spectrum,
            label="Initial guess",
        )
        ax[1].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[1].set_ylabel("Intensity (Arb. unit)")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].legend()

        if plot_xmin is not None and plot_xmax is not None:
            for a in ax:
                a.set_xlim(plot_xmin, plot_xmax)

        return fig


def generate_response_curve(
    wavelength: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    relative_wl = wavelength - coefficients[0]
    taylor_series = coefficients[1] + 0.5 * relative_wl * coefficients[2]
    gaussian = np.exp(
        -(np.abs(relative_wl) ** coefficients[4])
        / (2 * coefficients[3] ** coefficients[4])
    )
    return taylor_series * gaussian


def get_new_wavelength(wavelengths_micron, taylor_coefficients_micron):
    distance = wavelengths_micron - taylor_coefficients_micron[0]
    taylor_shift = taylor_coefficients_micron[1] * np.ones(distance.shape, dtype=float)
    for i in range(2, len(taylor_coefficients_micron)):
        taylor_shift += distance ** (i - 1) * taylor_coefficients_micron[i]

    new_wavelengths = 1e-6 * (wavelengths_micron + taylor_shift)
    new_freq = constants.speed_of_light / new_wavelengths
    return new_wavelengths, new_freq


def fit_calibration_amplitude_model(
    measurement, reference, wavelength_coeffs, amplitude_guess, roi
):
    initial_cal = generate_calibration_from_coeffs(
        amplitude_guess, wavelength_coeffs, measurement.wavelength
    )
    reference_shift = reference.to_interpolated_wavelength(
        initial_cal.corrected_wavelengths
    )

    def residual_amp(coeff):
        cal = generate_calibration_from_coeffs(
            coeff, wavelength_coeffs, measurement.wavelength
        )
        calibrated = cal.apply_to_spectrum(measurement)
        residuals = calibrated.spectrum - reference_shift.spectrum
        return residuals[
            (calibrated.wavelength > roi[0]) & (calibrated.wavelength < roi[1])
        ]

    amplitude_result = scipy.optimize.least_squares(
        residual_amp, amplitude_guess, ftol=1e-12, max_nfev=16384
    )

    return amplitude_result.x


def generate_calibration_from_coeffs(amplitude_coeffs, wavelength_coeffs, wavelengths):
    new_wavelengths, new_freqs = get_new_wavelength(
        wavelengths * 1e6, wavelength_coeffs
    )
    intensity_factors = 1.0 / generate_response_curve(
        wavelength=new_wavelengths * 1e6, coefficients=amplitude_coeffs
    )
    return SpectrometerCalibration(
        intensity_factors=intensity_factors,
        corrected_wavelengths=new_wavelengths,
        corrected_frequencies=new_freqs,
    )


@yaml_io
@dataclass
class CalibrationDataset:
    measurement: IntensitySpectrum
    reference: IntensitySpectrum
    input_parameters: CalibrationInput
    parameterized_calibration: SpectrometerCalibration
    measurement_with_parameterized_calibration: IntensitySpectrum
    residuals: np.ndarray
    final_calibration: SpectrometerCalibration

    def plot(self, plot_xmin=None, plot_xmax=None):
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        ax[0, 0].plot(
            self.measurement.wavelength_nm(),
            self.measurement.spectrum,
            label="Measurement",
        )
        ax[0, 0].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 0].set_xlabel("Wavelength (nm)")
        ax[0, 0].set_ylabel("Intensity (Arb. unit)")
        ax[0, 0].legend()
        initial_guess_calibration = generate_calibration_from_coeffs(
            self.input_parameters.get_amplitude_array(),
            self.input_parameters.get_wavelength_array(),
            self.measurement.wavelength,
        )
        initial_guess_spectrum = initial_guess_calibration.apply_to_spectrum(
            self.measurement
        )
        ax[0, 1].plot(
            initial_guess_spectrum.wavelength_nm(),
            initial_guess_spectrum.spectrum,
            label="Initial guess",
        )
        ax[0, 1].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 1].set_ylabel("Intensity (Arb. unit)")
        ax[0, 1].set_xlabel("Wavelength (nm)")
        ax[0, 1].legend()

        ax[0, 2].plot(
            self.measurement_with_parameterized_calibration.wavelength_nm(),
            self.measurement_with_parameterized_calibration.spectrum,
            label="Model calibrated",
        )
        ax[0, 2].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 2].set_ylabel("Intensity (Arb. unit)")
        ax[0, 2].set_xlabel("Wavelength (nm)")
        ax[0, 2].legend()

        ax[1, 0].plot(
            self.measurement_with_parameterized_calibration.wavelength_nm(),
            self.residuals,
            label="Residual",
        )
        ax[1, 0].set_ylim(-0.1, 0.1)
        ax[1, 0].set_ylabel("Intensity (Arb. unit)")
        ax[1, 0].set_xlabel("Wavelength (nm)")
        ax[1, 0].legend()

        second_calibration = self.final_calibration.apply_to_spectrum(self.measurement)
        ax[1, 1].plot(
            second_calibration.wavelength_nm(),
            second_calibration.spectrum,
            label="Calibrated",
        )
        ax[1, 1].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[1, 1].set_ylabel("Intensity (Arb. unit)")
        ax[1, 1].set_xlabel("Wavelength (nm)")
        ax[1, 1].legend()
        ax[1, 2].plot(
            self.final_calibration.corrected_wavelengths * 1e9,
            self.final_calibration.intensity_factors,
            label="Final weights",
        )
        ax[1, 2].set_ylabel("Final weights")
        ax[1, 2].set_xlabel("Wavelength (nm)")
        ax[1, 2].legend()

        if plot_xmin is not None and plot_xmax is not None:
            for a in ax:
                for b in a:
                    b.set_xlim(plot_xmin, plot_xmax)

        return fig

    @staticmethod
    def generate(
        measurement: IntensitySpectrum,
        reference: IntensitySpectrum,
        input_parameters: CalibrationInput,
    ):
        new_guess = fit_calibration_amplitude_model(
            measurement=measurement,
            reference=reference,
            wavelength_coeffs=input_parameters.get_wavelength_array(),
            amplitude_guess=input_parameters.get_amplitude_array(),
            roi=np.array([input_parameters.roi_lowest, input_parameters.roi_highest]),
        )

        parameterized_calibration = generate_calibration_from_coeffs(
            new_guess, input_parameters.get_wavelength_array(), measurement.wavelength
        )
        measurement_with_parameterized_calibration = (
            parameterized_calibration.apply_to_spectrum(measurement)
        )

        ref_on_new_axis = reference.to_interpolated_wavelength(
            measurement_with_parameterized_calibration.wavelength
        )
        measurement_shifted = measurement.to_corrected_wavelength(
            measurement_with_parameterized_calibration.wavelength
        )
        residuals = (
            measurement_with_parameterized_calibration.spectrum
            - ref_on_new_axis.spectrum
        )

        new_weights = (
            parameterized_calibration.intensity_factors
            - measurement_shifted.spectrum
            * residuals
            / (measurement_shifted.spectrum**2 + input_parameters.noise_level)
        )

        final_calibration = SpectrometerCalibration(
            intensity_factors=new_weights,
            corrected_frequencies=parameterized_calibration.corrected_frequencies,
            corrected_wavelengths=parameterized_calibration.corrected_wavelengths,
        )

        return CalibrationDataset(
            measurement=measurement,
            reference=reference,
            input_parameters=input_parameters,
            parameterized_calibration=parameterized_calibration,
            measurement_with_parameterized_calibration=measurement_with_parameterized_calibration,
            residuals=residuals,
            final_calibration=final_calibration,
        )
