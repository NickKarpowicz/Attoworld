import numpy as np
import matplotlib.pyplot as plt
import pandas

class FrogResult:
    """Loads the FROG reconstructed spectra and traces together with the measured spectrogram
    
    files are in the format:
    filename.A.dat: measured spectrogram
    filename.Arecon.dat: reconstructed spectrogram
    filename.Ek.dat: field envelope, phase
    filename.Speck.dat: reconstructed spectrum and phase

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

        fig, ax = plt.subplots(figsize=[6.4, 4.8])
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
        ax2.plot(limited_wvl, limited_phase, label='direct phase from file', color='r')
        ax2.plot(limited_wvl, limited_phase_r_i, label='phase from re/im FFT field', color='g')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('amplitude (a.u.)')
        ax2.set_ylabel('phase (rad)', color='r')
        ax2.tick_params(axis='y', colors='r')
        ax2.yaxis.label.set_color('r')
        plt.show()

    def test_FROG_result_format_envelope(self, low_lim=None, up_lim=None):
        """checks (by plotting) that the quantities stored in the file are the expected ones"""

        fig, ax = plt.subplots(figsize=[6.4, 4.8])
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
        ax2.plot(limited_time, limited_tphase, label='direct phase from file', color='r')
        ax2.plot(limited_time, limited_tphase_r_i, label='phase from re/im field', color='g')
        ax.set_xlabel('time (fs)')
        ax.set_ylabel('amplitude (a.u.)')
        ax2.set_ylabel('phase (rad)', color='r')
        ax2.tick_params(axis='y', colors='r')
        ax2.yaxis.label.set_color('r')
        plt.show()

    def get_squared_envelope(self):
        """returns time (fs), intensity envelope of the reconstructed pulse (square of the field envelope)"""
        return self.time, self.envelope

    def get_FWHM(self):
        tfine = np.linspace(self.time[0], self.time[-1], len(self.time)*20)
        enSquare = np.interp(tfine ,self.time, self.envelope)
        i_max = np.argmax(enSquare)
        hm = enSquare[i_max] / 2
        fwhm = (np.max(tfine[np.argwhere(enSquare >= hm)]) - np.min(tfine[np.argwhere(enSquare >= hm)]))
        return fwhm

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

    def plot_temporal_profile(self, low_lim = None, up_lim = None):
        fig, ax = plt.subplots(figsize=[6.4, 4.8])
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        ax.set_title('Reconstruction')
        ax.plot(self.time[(self.time>low_lim)&(self.time<up_lim)], self.envelope[(self.time>low_lim)&(self.time<up_lim)])
        ax2.plot(self.time[(self.time>low_lim)&(self.time<up_lim)], self.tphase[(self.time>low_lim)&(self.time<up_lim)], color='r')
        ax.set_xlabel('time (fs)')
        ax.set_ylabel('intensity envelope (a.u.)')
        ax2.set_ylabel('phase (rad)', color='r')
        ax2.tick_params(axis='y', colors='r')
        ax2.yaxis.label.set_color('r')
        plt.show()

    def plot_spectrum(self, low_lim = None, up_lim = None):
        fig, ax = plt.subplots(figsize=[6.4, 4.8])
        if low_lim is not None and up_lim is not None:
            ax.set_xlim(low_lim,up_lim)
        ax2 = ax.twinx()
        ax.set_title('Reconstruction')
        ax.plot(self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)], self.spectrum[(self.wvl>low_lim)&(self.wvl<up_lim)]*(1/self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)]**2))
        ax2.plot(self.wvl[(self.wvl>low_lim)&(self.wvl<up_lim)], self.sphase[(self.wvl>low_lim)&(self.wvl<up_lim)], color='r')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('amplitude (a.u.)')
        ax2.set_ylabel('phase (rad)', color='r')
        ax2.tick_params(axis='y', colors='r')
        ax2.yaxis.label.set_color('r')
        plt.show()

    def plot_spectrograms(self, low_lim_t = None, up_lim_t = None, low_lim_f = None, up_lim_f = None):
        fig, ax = plt.subplots(figsize=[6.4, 4.8])
        ax.set_title('Reconstruction')
        X, Y = np.meshgrid(self.rec_taxis, self.rec_faxis)
        ax.pcolormesh(X, Y * 1.e-3, self.reconstruct)
        ax.set_xlabel('time (fs)')
        ax.set_ylabel('frequency (PHz)')
        if low_lim_t is not None and up_lim_t is not None:
            ax.set_xlim(low_lim_t, up_lim_t)
        if low_lim_f is not None and up_lim_f is not None:
            ax.set_ylim(low_lim_f, up_lim_f)
        plt.show()

        fig, ax = plt.subplots(figsize=[6.4, 4.8])
        ax.set_title('Measurement')
        X, Y = np.meshgrid(self.meas_taxis, self.meas_faxis)
        print(self.meas_taxis.shape, X.shape, self.measurement.shape)
        ax.pcolormesh(X, Y * 1.e-3, self.measurement)
        ax.set_xlabel('time (fs)')
        ax.set_ylabel('frequency (PHz)')
        if low_lim_t is not None and up_lim_t is not None:
            ax.set_xlim(low_lim_t, up_lim_t)
        if low_lim_f is not None and up_lim_f is not None:
            ax.set_ylim(low_lim_f, up_lim_f)
        plt.show()



