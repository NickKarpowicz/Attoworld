import numpy as np
import matplotlib.pyplot as plt
import pandas
from ..numeric import fwhm, interpolate, derivative
from ..plot import label_letter
from dataclasses import dataclass
from scipy import constants
from .dataclasses import ComplexSpectrum, Spectrogram, Waveform
from pathlib import Path
from matplotlib.axes import Axes
from typing import Optional

def read_Trebino_FROG_matrix(filename: Path | str) -> Spectrogram:
    """
    Read a spectrogram file made by the Trebino FROG code

    Args:
        filename (Path | str): the name (path) of the file
    """
    with open(filename, "r") as f:
        l = str(f.readline())
        l = l.split()
        n1 = int(l[0])
        n2 = int(l[1])
        l = str(f.readline())
        l = l.split()
    measured_data = pandas.read_csv(filename, sep='\t', header = None, skiprows=2)
    measure = []
    raw_freq = 1e9*constants.speed_of_light/np.array(measured_data[0][0:n2]).squeeze()
    df = np.mean(np.diff(raw_freq))
    freq = raw_freq[0] + df * np.array(range(raw_freq.shape[0]))
    time = 1e-15 * np.array(measured_data[0][n2:(n2+n1)]).squeeze()
    for i in range(n1):
        measure.append(measured_data[0][(i+2)*n2:(i+3)*n2])
    data = np.array(measure)
    return Spectrogram(data = data, time = time, freq = freq)

def read_Trebino_FROG_speck(filename: Path | str) -> ComplexSpectrum:
    """
    Read a .Speck file made by the Trebino FROG code

    Args:
        filename (Path | str): the name (path) of the file
    """
    data = np.array(pandas.read_csv(filename, sep='\t', header = None), dtype=float)
    raw_freq = 1e9*constants.speed_of_light/data[:,0]
    df = np.mean(np.diff(raw_freq))
    freq = np.linspace(0.0, raw_freq[-1], int(np.ceil(raw_freq[-1]/df)))
    spectrum = interpolate(freq, raw_freq, data[:,3]) + 1j * interpolate(freq, raw_freq, data[:,4])
    return ComplexSpectrum(spectrum=spectrum, freq=freq)

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
        a = ax.pcolormesh(
            1e15 * self.measured_spectrogram.time,
            1e-12 * self.measured_spectrogram.freq,
            self.measured_spectrogram.data / np.max(self.measured_spectrogram.data[:]),
            rasterized=True)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Frequency (THz)')
        ax.set_title('Measurement')
        plt.colorbar(a)
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
        a=ax.pcolormesh(
            1e15 * self.reconstructed_spectrogram.time,
            1e-12 * self.reconstructed_spectrogram.freq,
            self.reconstructed_spectrogram.data/np.max(self.reconstructed_spectrogram.data[:]),
            rasterized=True)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Frequency (THz)')
        ax.set_title(f"Reconstruction (G': {self.get_error():0.4f})")
        plt.colorbar(a)
        return fig

    def plot_pulse(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """
        Plot the reconstructed pulse.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (instantaneous frequency) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        envelope = self.pulse.to_complex_envelope().envelope
        if envelope is not None:
            time_ax = self.pulse.time_fs()-np.mean(self.pulse.time_fs())
            intensity = np.abs(envelope)**2
            intensity /= np.max(intensity)
            intensity_line = ax.plot(time_ax,
                intensity,
                label=f"Intensity, fwhm {1e15*self.get_fwhm():0.1f} fs")
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Intensity (Arb. unit)')
            ax_phase = plt.twinx(ax)
            inst_freq = (1e-12/(2*np.pi))*derivative(np.unwrap(np.angle(envelope)), 1)/self.pulse.dt
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
            ax.legend(lines, [l.get_label() for l in lines])
        return fig
    def plot_spectrum(self, ax: Optional[Axes] = None, phase_blanking: float = 0.05, xlim=None):
        """
        Plot the reconstructed spectrum and group delay curve.

        Args:
            ax: optionally plot onto a pre-existing matplotlib Axes
            phase_blanking: only show phase information (group delay) above this level relative to max intensity
            xlim: pass arguments to set_xlim() to constrain the x-axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        spec = self.spectrum.to_intensity_spectrum()
        intensity = spec.spectrum
        phase = spec.phase
        freq = spec.freq
        if intensity is not None and phase is not None and freq is not None:
            start_index = np.argmax(intensity>0)
            print(start_index)
            intensity = intensity[start_index::]
            freq = freq[start_index::]
            wl = constants.speed_of_light/freq
            phase = np.unwrap(phase[start_index::])

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
            ax.legend(lines, [l.get_label() for l in lines])
        return fig

    def plot_all(self, phase_blanking=0.05, time_xlims=None, wavelength_xlims=None):
        """
        Produce a 4-panel plot of the FROG results, combining calls to plot_measured_spectrogram(),
        plot_reconstructed_spectrogram(), plot_pulse() and plot_spectrum() as subplots, with letter labels.

        Args:
            phase_blanking: relative intensity at which to show phase information
            time_xlim: x-axis limits to pass to the plot of the pulse
            wavelength_xlim: x-axis limits to pass to the plot of the spectrum"""
        default_figsize = plt.rcParams['figure.figsize']
        fig,ax = plt.subplots(2,2, figsize=(default_figsize[0] * 2, default_figsize[1]*2))
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
        return np.sqrt(
            np.sum( (self.measured_spectrogram.data[:] - self.reconstructed_spectrogram.data[:])**2)
            / np.sum(self.measured_spectrogram.data[:]**2))
    def get_fwhm(self) -> float:
        """
        Get the full-width-at-half-max value of the reconstructed pulse
        """
        return self.pulse.get_envelope_fwhm()

def read_Trebino_FROG_data(filename: str) -> FrogData:
    """
    Read a set of data produced by the Trebino FROG reconstruction code.

    Args:
        filename: Base filename of the .bin file; e.g. if the data are mydata.bin.Speck.dat etc., this will be "mydata.bin" """
    spectrum = read_Trebino_FROG_speck(filename+'.Speck.dat')
    pulse = spectrum.to_centered_waveform()
    measured_spectrogram = read_Trebino_FROG_matrix(filename+'.A.dat')
    reconstructed_spectrogram = read_Trebino_FROG_matrix(filename+'.Arecon.dat')
    return FrogData(spectrum = spectrum,
        pulse = pulse,
        measured_spectrogram = measured_spectrogram,
        reconstructed_spectrogram = reconstructed_spectrogram)

class FrogResult:
    """Loads the FROG reconstructed spectra and traces together with the measured spectrogram

    Files are in the format (all of them need to be available):
    * filename.A.dat: measured spectrogram
    * filename.Arecon.dat: reconstructed spectrogram
    * filename.Ek.dat: field envelope, phase
    * filename.Speck.dat: reconstructed spectrum and phase

    Warning: the class stores the data as in the files, please use methods "get_*()" to access the data correctly.
    """
    def __init__(self, filename):
        # reconstructed (spectral domain)
        self.wvl = None
        self.spectrum = None    # !!! PSEUDO-SPECTRUM, NOT multiplied by 1/λ^2
        self.sphase = None
        self.realFFT = None     # named realFFT but expressed as a function of wavelength
        self.imagFFT = None

        # reconstructed (temporal domain)
        self.time = None
        self.envelope = None    # INTENSITY ENVELOPE (square of the field envelope)
        self.tphase = None
        self.realField = None
        self.imagField = None

        #spectrograms
        self.meas_taxis = None
        self.meas_faxis = None
        self.rec_taxis = None
        self.rec_faxis = None
        self.measurement = None
        self.reconstruct = None

        self.load(filename)

    def test_FROG_result_format_spectrum(self, low_lim=None, up_lim=None):
        """checks (by plotting) that the quantities stored in the file are the expected ones"""

        fig, ax = plt.subplots()
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        ax.set_title('(should be 2 identic. spectra + 2 identic. phases)')
        limited_wvl = self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)]
        limited_spectrum = self.spectrum[(self.wvl>low_lim)&(self.wvl<up_lim)]
        limited_spectrum_r_i = (self.realFFT[(self.wvl>low_lim)&(self.wvl<up_lim)]**2 + self.imagFFT[(self.wvl>low_lim)&(self.wvl<up_lim)]**2)
        limited_phase = self.sphase[(self.wvl>low_lim)&(self.wvl<up_lim)]
        limited_phase_r_i = np.arctan2(self.imagFFT[(self.wvl>low_lim)&(self.wvl<up_lim)], self.realFFT[(self.wvl>low_lim)&(self.wvl<up_lim)])
        limited_phase_r_i = -np.unwrap(limited_phase_r_i)
        ax.plot(limited_wvl, limited_spectrum/np.max(limited_spectrum), label='direct spectrum from file')
        ax.plot(limited_wvl, limited_spectrum_r_i/np.max(limited_spectrum_r_i), label='sp. from re/im FFT field')
        ax2.plot([],[])
        ax2.plot([],[])
        ax2.plot(limited_wvl, limited_phase, label='direct phase from file')
        ax2.plot(limited_wvl, limited_phase_r_i, label='phase from re/im FFT field')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Amplitude (Arb. unit)')
        ax2.set_ylabel('Phase (rad)')
        ax2.tick_params(axis='y')

    def test_FROG_result_format_envelope(self, low_lim=None, up_lim=None):
        """checks (by plotting) that the quantities stored in the file are the expected ones"""

        fig, ax = plt.subplots()
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        ax.set_title('(should be 2 identic. envelopes + 2 ident. phases)')
        limited_time = self.time[(self.time>low_lim)&(self.time<up_lim)]
        limited_envelope = self.envelope[(self.time>low_lim)&(self.time<up_lim)]
        limited_envelope_r_i = (self.realField[(self.time>low_lim)&(self.time<up_lim)]**2 + self.imagField[(self.time>low_lim)&(self.time<up_lim)]**2)
        limited_tphase = self.tphase[(self.time>low_lim)&(self.time<up_lim)]
        limited_tphase_r_i = np.arctan2(self.imagField[(self.time>low_lim)&(self.time<up_lim)], self.realField[(self.time>low_lim)&(self.time<up_lim)])
        limited_tphase_r_i = -np.unwrap(limited_tphase_r_i)
        ax.plot(limited_time, limited_envelope/np.max(limited_envelope), label='direct env. from file')
        ax.plot(limited_time, limited_envelope_r_i/np.max(limited_envelope_r_i), label='sp. from re/im field')
        ax2.plot([],[])
        ax2.plot([],[])
        ax2.plot(limited_time, limited_tphase, label='direct phase from file')
        ax2.plot(limited_time, limited_tphase_r_i, label='phase from re/im field')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Amplitude (Arb. unit)')
        ax2.set_ylabel('Phase (rad)')

    def get_squared_envelope(self):
        """returns time (fs), intensity envelope of the reconstructed pulse (square of the field envelope)"""
        return self.time, self.envelope

    def get_FWHM(self):
        return fwhm(self.envelope, self.time[1] - self.time[2])


    def get_spectral_phase(self):
        """returns wavelength in nm, reconstructed spectral phase in rad"""
        return self.wvl, self.sphase

    def get_spectrum(self):
        """returns wavelength in nm, reconstructed intensity spectrum (a.u.)"""
        return self.wvl, self.spectrum * 1/self.wvl**2

    def get_temporal_phase(self):
        """returns time in fs, reconstructed temporal phase in rad"""
        return self.time, self.tphase

    def load(self, filename):
        """content of filename.Ek.dat (columns):
         - time (fs)
         - intensity envelope (square of field envelope)
         - instantaneous phase (rad) (net of the carrier phase, and with a minus sign)
         - real part of the field envelope (net of the carrier frequency oscillations)
         - imaginary part of the field envelope (net of the carrier frequency oscillations)

         content of filename.Speck.dat:
         - wavelengths (nm)
         - pseudo-intensity spectrum |FFT|^2  (NOT multiplied by 1/λ^2)
         - spectral phase (rad) (with a minus sign
         - real part of FFT field (experessed as a function of wavelength, of course)
         - imaginary part of FFT field (expressed as a function of wavelength, of course)

         content of spectrogram files:
         1st line: nbins x    nbins y
         2nd line: ? (unkn.)  ? (unkn.) (probably G error and Z error of the reconstruction)
         single column: concatenation of
            - frequency axis (THz)
            - time axis (fs)
            - delay dependent spectral data"""

        data_field = pandas.read_csv(filename + ".Ek.dat", sep='\t', header = None)
        data_spec = pandas.read_csv(filename + ".Speck.dat", sep='\t', header = None)
        with open(filename + ".A.dat", "r") as f:
            l = str(f.readline())
            l = l.split()
            n1 = int(l[0])
            n2 = int(l[1])
            l = str(f.readline())
            l = l.split()
            e1 = float(l[0])
            e2 = float(l[1])
        measured_data = pandas.read_csv(filename + ".A.dat", sep='\t', header = None, skiprows=2)
        measure = []
        self.meas_faxis = measured_data[0][0:n2]
        self.meas_taxis = measured_data[0][n2:(n2+n1)]
        for i in range(n1):
            measure.append(measured_data[0][(i+2)*n2:(i+3)*n2])
        self.measurement = np.array(measure)

        with open(filename + ".Arecon.dat", "r") as f:
            l = str(f.readline())
            l = l.split()
            n1 = int(l[0])
            n2 = int(l[1])
            l = str(f.readline())
            l = l.split()
            e1 = float(l[0])
            e2 = float(l[1])
        reconstructed_data = pandas.read_csv(filename + ".Arecon.dat", sep='\t', header = None, skiprows=2)
        reconstruct = []
        self.rec_faxis = reconstructed_data[0][0:n2]
        self.rec_taxis = reconstructed_data[0][n2:(n2+n1)]
        for i in range(n1):
            reconstruct.append(reconstructed_data[0][(i+2)*n2:(i+3)*n2])
        self.reconstruct = np.array(reconstruct)

        self.time = np.array(data_field[0])
        self.envelope = np.array(data_field[1])
        self.tphase = np.array(data_field[2])
        self.realField = np.array(data_field[3])
        self.imagField = np.array(data_field[4])

        self.wvl = np.array(data_spec[0])
        self.spectrum = np.array(data_spec[1])
        self.sphase = np.array(data_spec[2])
        self.realFFT = np.array(data_spec[3])
        self.imagFFT = np.array(data_spec[4])
        # remove negative wavelengths
        self.spectrum = self.spectrum[self.wvl>0]
        self.sphase = self.sphase[self.wvl>0]
        self.realFFT = self.realFFT[self.wvl>0]
        self.imagFFT = self.imagFFT[self.wvl>0]
        self.wvl = self.wvl[self.wvl>0]

    def plot_temporal_profile(self, low_lim = None, up_lim = None, phase_blanking_level=0.05):
        fig, ax = plt.subplots()
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        min_phase_intensity = phase_blanking_level * np.max(self.envelope)
        intensity_line = ax.plot(self.time[(self.time>low_lim)&(self.time<up_lim)],
            self.envelope[(self.time>low_lim)&(self.time<up_lim)],
            label='Intensity')
        ax2.plot([],[])
        phase_line = ax2.plot(self.time[(self.time>low_lim)&(self.time<up_lim)&(self.envelope>min_phase_intensity)],
            self.tphase[(self.time>low_lim)&(self.time<up_lim)&(self.envelope>min_phase_intensity)],
            '--',label='Phase')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Intensity envelope (Arb. unit)')
        ax2.set_ylabel('Phase (rad)')
        lines = intensity_line+phase_line
        ax.legend(lines, [l.get_label() for l in lines])
        return fig

    def plot_spectrum(self, low_lim = None, up_lim = None, phase_blanking_level=0.05):
        fig, ax = plt.subplots()
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        min_phase_intensity = phase_blanking_level * np.max(self.spectrum)
        intensity_line = ax.plot(self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)], self.spectrum[(self.wvl>low_lim)&(self.wvl<up_lim)]*(1/self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)]**2), label="Intensity")
        ax2.plot([],[])
        phase_line = ax2.plot(self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)&(self.spectrum>min_phase_intensity)], self.sphase[(self.wvl>low_lim)&(self.wvl<up_lim)&(self.spectrum>min_phase_intensity)], '--', label="Phase")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Amplitude (Arb. unit)')
        ax2.set_ylabel('Phase (rad)')
        lines = intensity_line+phase_line
        ax.legend(lines, [l.get_label() for l in lines])
        return fig

    def plot_spectrograms(self, low_lim_t = None, up_lim_t = None, low_lim_f = None, up_lim_f = None, letter_style='Nature'):
        original_figsize = plt.rcParams.get('figure.figsize')
        double_figsize = (original_figsize[0],2*original_figsize[1])
        fig, ax = plt.subplots(2,1,figsize=double_figsize)
        X, Y = np.meshgrid(self.rec_taxis, self.rec_faxis)
        ax[1].pcolormesh(X, Y * 1.e-3, self.reconstruct, rasterized=True)
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('Frequency (PHz)')
        if low_lim_t is not None and up_lim_t is not None:
            ax[1].set_xlim(low_lim_t, up_lim_t)
        if low_lim_f is not None and up_lim_f is not None:
            ax[1].set_ylim(low_lim_f, up_lim_f)
        label_letter('a',axis=ax[0], style=letter_style)

        X, Y = np.meshgrid(self.meas_taxis, self.meas_faxis)
        print(self.meas_taxis.shape, X.shape, self.measurement.shape)
        ax[0].pcolormesh(X, Y * 1.e-3, self.measurement, rasterized=True)
        ax[0].set_xlabel('Time (fs)')
        ax[0].set_ylabel('Frequency (PHz)')
        if low_lim_t is not None and up_lim_t is not None:
            ax[0].set_xlim(low_lim_t, up_lim_t)
        if low_lim_f is not None and up_lim_f is not None:
            ax[0].set_ylim(low_lim_f, up_lim_f)
        label_letter('b',axis=ax[1], style=letter_style)
        plt.rcParams.update({'figure.figsize': original_figsize})
        return fig
