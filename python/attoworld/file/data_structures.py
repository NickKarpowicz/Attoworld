from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..numeric import interpolate, fwhm, find_maximum_location, derivative
from ..plot import label_letter
from ..spectrum import wavelength_to_frequency, frequency_to_wavelength, transform_limited_pulse_from_spectrometer
from scipy import constants
import scipy.signal as sig
import copy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
        with open(filename, 'w') as file:
            file.write(f'{len(self.freq)}\t{len(self.time)}\n')
            file.write(f'{np.min(self.data[:])}\t{np.max(self.data[:])}\n')
            for freq in self.freq:
                wavelength_nm = 1e9*constants.speed_of_light/freq
                file.write(f'{wavelength_nm:15.15g}\n')
            for time in self.time:
                time_fs = 1e15 * time
                file.write(f'{time_fs:15.15g}\n')
            for x in self.data:
                for y in x:
                    file.write(f'{y:15.15g}\n')

    def to_per_frequency_dc_removed(self, extra_offset: float = 0.0):
        """Perform DC offset removal on a measured spectrogram, on a per-frequency basis

        Args:
            extra_offset (float): subtract a value from the entire array (negative values are always set to zero)

        Returns:
            Spectrogram: the spectrogram with offset removed."""
        new_data = np.array(self.data)
        new_data -= extra_offset
        new_data[new_data<0.0] = 0.0
        for _i in range(new_data.shape[0]):
            new_data[_i,:] -= np.min(new_data[_i,:])

        return Spectrogram(data = new_data, time=self.time, freq=self.freq)
    def to_symmetrized(self):
        """
        Average the trace with a time-reversed copy. This might be useful for getting a reconstruction of difficult data, but keep in mind that the resulting measured trace will no longer represent the real measurement and should not be published as such.
        """
        return Spectrogram(data = 0.5 * (self.data + np.fliplr(self.data)), time = self.time, freq = self.freq)
    def to_binned(self, dim: int = 64, dt: float = 5e-15, t0: Optional[float] = None, f0: float = 750e12):
        """Bin a spectrogram to a FFT-appropriate shape

        Args:
            dim (int): size of each size of the resulting square data
            dt (float): time step of the data
            t0: (Optional[float]): time-zero of the data. If not specified, will be calculated by the first moment of the time-distribution of the signal
            f0: (float): central frequency of the binned array

        Returns:
            Spectrogram: the binned spectrogram
        """
        _t = np.array(range(dim))*dt
        _t -= np.mean(_t)
        _f = np.fft.fftshift(np.fft.fftfreq(dim, d=dt) + f0)
        binned_data = np.zeros((dim,self.time.shape[0]),dtype=float)
        for _i in range(self.time.shape[0]):
            binned_data[:,_i] = interpolate(_f, self.freq, np.array(self.data[:,_i]), neighbors=2)
        binned_data /= np.max(binned_data[:])
        if t0 is None:
            ac = np.sum(binned_data,axis=0)
            t0 = np.sum(ac*self.time)/np.sum(ac)
        binned_data_square = np.zeros((dim,dim),dtype=float)
        for _i in range(dim):
            binned_data_square[_i,:] = interpolate(_t, self.time-t0, np.array(binned_data[_i,:]), neighbors=2)
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
        a=ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            self.data/np.max(self.data[:]),
            rasterized=True)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Frequency (THz)')
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
        logdata[self.data>0.0] = np.log(self.data[self.data>0.0])
        logdata[self.data<0.0] = 0.0
        a=ax.pcolormesh(
            1e15 * self.time,
            1e-12 * self.freq,
            logdata,
            rasterized=True)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Frequency (THz)')
        ax.grid(True,lw=1)
        plt.colorbar(a)
        return fig
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
        """
        Create a Waveform based on this complex spectrum and center it in the time window
        """
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

    def plot_with_group_delay(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
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

        if self.spectrum is not None and self.phase is not None and self.freq is not None:
            start_index = np.argmax(self.spectrum>0)
            intensity = self.spectrum[start_index::]
            freq = self.freq[start_index::]
            wl = constants.speed_of_light/freq
            phase = np.unwrap(self.phase[start_index::])

            intensity /= np.max(intensity)
            intensity_line = ax.plot(1e9*wl,
                intensity,
                label="Intensity")
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity (Arb. unit)')
            ax_phase = plt.twinx(ax)
            group_delay = (1e15/(2*np.pi))*derivative(phase, 1)/(freq[1]-freq[2])
            ax_phase.plot([],[])
            phase_line = ax_phase.plot(
                1e9*wl[intensity>phase_blanking],
                group_delay[intensity > phase_blanking],
                '--',
                label='Group delay')
            ax_phase.set_ylabel('Group delay (fs)')
            if xlim is not None:
                ax.set_xlim(xlim)
                ax_phase.set_xlim(xlim)
            lines = lines = intensity_line+phase_line
            ax.legend(lines, [line.get_label() for line in lines])
        return fig
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
    def time_fs(self):
        if self.time is not None:
            return 1e15*self.time
        else:
            raise Exception("No time axis.")
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
        envelope = self.envelope
        if envelope is not None:
            time_ax = self.time_fs()-np.mean(self.time_fs())
            intensity = np.abs(envelope)**2
            intensity /= np.max(intensity)
            intensity_line = ax.plot(time_ax,
                intensity,
                label=f"Intensity, fwhm {1e15*self.get_fwhm():0.1f} fs")
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Intensity (Arb. unit)')
            ax_phase = plt.twinx(ax)
            inst_freq = (1e-12/(2*np.pi))*derivative(np.unwrap(np.angle(envelope)), 1)/self.dt
            ax_phase.plot([],[])
            phase_line = ax_phase.plot(
                time_ax[intensity>phase_blanking],
                inst_freq[intensity > phase_blanking],
                '--',
                label='Inst. frequency')
            ax_phase.set_ylabel('Inst. frequency (THz)')
            if xlim is not None:
                ax.set_xlim(xlim)
                ax_phase.set_xlim(xlim)
            lines = lines = intensity_line+phase_line
            ax.legend(lines, [line.get_label() for line in lines])
        return fig

@dataclass(frozen=True, slots=True)
class FrogData:
    """
    Stores data from a FROG measurement

    Attributes:
        spectrum (ComplexSpectrum): the reconstructed complex spectrum
        pulse (Waveform): time-domain reconstructed field
        measured_spectrogram (Spectrogram): measured (binned) data
        reconstructed_spectrogram (Spectrogram): spectrogram resulting from reconstructed field
    """
    spectrum: ComplexSpectrum
    pulse: Waveform
    measured_spectrogram: Spectrogram
    reconstructed_spectrogram: Spectrogram
    raw_reconstruction: np.ndarray
    f0: float
    dt: float

    def save(self, base_filename):
        """
        Save in the Trebino FROG format

        Args:
            base_filename: base of the file path; 4 files will be made from it: .A.dat, .Arecon.dat, .Ek.dat, and .Speck.dat
        """
        self.measured_spectrogram.save(base_filename+'.A.dat')
        self.reconstructed_spectrogram.save(base_filename+'.Arecon.dat')

        t = 1e15 * self.dt * np.array(range(len(self.raw_reconstruction)))
        t -= np.mean(t)
        f = np.fft.fftshift(np.fft.fftfreq(len(t),d=self.dt)) + self.f0
        lam = constants.speed_of_light/f
        raw_spec = np.fft.fftshift(np.fft.fft(self.raw_reconstruction))

        with open(base_filename+'.Ek.dat','w') as time_file:
            for _i in range(len(self.raw_reconstruction)):
                time_file.write(f'{t[_i]:.15g}\t{np.abs(self.raw_reconstruction[_i])**2:.15g}\t{np.angle(self.raw_reconstruction[_i]):.15g}\t{np.real(self.raw_reconstruction[_i]):.15g}\t{np.imag(self.raw_reconstruction[_i]):.15g}\n')

        with open(base_filename+'.Speck.dat','w') as spec_file:
            for _i in range(len(self.raw_reconstruction)):
                spec_file.write(f'{lam[_i]:.15g}\t{np.abs(raw_spec[_i])**2:.15g}\t{np.angle(raw_spec[_i]):.15g}\t{np.real(raw_spec[_i]):.15g}\t{np.imag(raw_spec[_i]):.15g}\n')

    def plot_measured_spectrogram(self, ax: Optional[Axes] = None):
        """
        Plot the measured spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self.measured_spectrogram.plot(ax)
        ax.set_title('Measurement')
        return fig

    def plot_reconstructed_spectrogram(self, ax: Optional[Axes] = None):
        """
        Plot the reconstructed spectrogram.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self.reconstructed_spectrogram.plot(ax)
        ax.set_title(f"Reconstruction (G': {self.get_error():0.2e})")
        return fig

    def plot_pulse(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """
        Plot the reconstructed pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        return self.pulse.to_complex_envelope().plot(ax, phase_blanking, xlim)
    def plot_spectrum(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """
        Plot the reconstructed spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        return self.spectrum.to_intensity_spectrum().plot_with_group_delay(ax, phase_blanking, xlim)

    def plot_all(self, phase_blanking=0.05, time_xlims=None, wavelength_xlims=None, figsize=None):
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
            default_figsize = plt.rcParams['figure.figsize']
            figsize = (default_figsize[0] * 2, default_figsize[1]*2)
        fig,ax = plt.subplots(2,2, figsize=figsize)
        self.plot_measured_spectrogram(ax[0,0])
        label_letter('a', ax[0,0])
        self.plot_reconstructed_spectrogram(ax[1,0])
        label_letter('b', ax[1,0])
        self.plot_pulse(ax[0,1], xlim=time_xlims)
        label_letter('c', ax[0,1])
        self.plot_spectrum(ax[1,1], xlim=wavelength_xlims)
        label_letter('d', ax[1,1])
        return fig
    def get_error(self) -> float:
        """
        Get the G' error of the reconstruction
        """
        norm_measured = np.linalg.norm(self.measured_spectrogram.data)
        norm_retrieved = np.linalg.norm(self.reconstructed_spectrogram.data)
        return np.sqrt(

            np.sum( (self.measured_spectrogram.data[:]/norm_measured - self.reconstructed_spectrogram.data[:]/norm_retrieved)**2) / np.sum((self.measured_spectrogram.data[:]/norm_measured)**2))
    def get_fwhm(self) -> float:
        """
        Get the full-width-at-half-max value of the reconstructed pulse
        """
        return self.pulse.get_envelope_fwhm()
