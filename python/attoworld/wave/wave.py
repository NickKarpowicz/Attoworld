import numpy as np
import scipy.signal as sig
import scipy.optimize as opt

def align_waves(waves, dt: float, frequency_roi_start: float, frequency_roi_stop: float):
    """
    Align a set of waveforms, inside of a 2D numpy array.
    Args:
        waves: set of waveforms
        dt (float): time step (assuming constant spacing), (s)
        frequency_roi_start (float): start frequency of region of interest for alignment (Hz)
        frequency_roi_stop (float): stop frequency of region of interest for alignment (Hz)

    Returns:
        np.ndarray: set of aligned waves
    """
    # Contributed by Nick Karpowicz
    waves_f = np.array(waves)

    #windowing and offset removal
    for i in range(waves_f.shape[0]):
        waves_f[i,:] = sig.windows.tukey(waves_f.shape[1]) * (waves_f[i,:] - np.mean(waves_f[i,:]))

    #find frequency roi
    f = np.fft.fftfreq(waves.shape[1],dt)
    f0 = np.argmin(np.abs(f-frequency_roi_start))
    f1 = np.argmin(np.abs(f-frequency_roi_stop))
    w_roi = 2*np.pi*f[f0:f1]

    #get active spectral region
    waves_f = np.fft.fft(waves, axis=1)
    waves_roi = np.array(waves_f[:,f0:f1])
    waves_roi /= np.max(np.max(np.abs(waves_roi)))

    #apply tau phase shifts
    def apply_taus(spectrum, taus, w):
        spectrum_shifted = np.zeros(spectrum.shape,dtype=complex)
        for i in range(spectrum.shape[0]):
            spectrum_shifted[i,:] = np.exp(-1j*1e-18*taus[i]*w)*spectrum[i,:]
        return spectrum_shifted

    #return fitting weights
    def get_residual(taus):
        shifted = apply_taus(waves_roi, taus, w_roi)
        mean_amplitudes = np.mean(shifted,axis=0)
        return 5.0-np.abs(mean_amplitudes)

    #apply fitting
    res = opt.least_squares(get_residual, np.zeros(waves.shape[0]), ftol = 1e-12, max_nfev=16384)

    #remove mean shift
    res.x -= np.mean(res.x)

    print(f"Rms shift in attoseconds: {np.std(res.x)}")
    return np.real(np.fft.ifft(apply_taus(waves_f,res.x,2*np.pi*f)))



def mean_offset_tukey(signal):
    """
    Apply DC offset removal and a Tukey window to a waveform

    Args:
        signal: the array describing the measured waveform

    Returns:
        np.ndarray: the modified signal
    """
    # Contributed by Nick Karpowicz
    mean_signal = np.mean(signal,axis=0)
    mean_signal -= np.mean(mean_signal)
    mean_signal *= sig.windows.tukey(mean_signal.shape[0])
    return mean_signal

def filtered_impulse_response(E_signal, E_reference, dt: float, filter_f0: float, filter_sigma: float, filter_order: int):
    """
    Calculate the filtered impulse response given by a signal and reference pair, with applied bandpass filter.

    This function takes two waveforms, E_signal and E_reference, and a bandpass filter, and returns the
    impulse response function of whatever caused the waveform to change

    Args:
        E_signal: the signal waveform array
        E_reference: the reference waveform
        dt (float): the time step
        filter_f0 (float): central frequency of the bandpass
        filter_sigma (float): width of the bandpass
        filter_order (int): order of the bandpass

    Returns:
        np.ndarray: the impuse response
    """
    # Contributed by Nick Karpowicz
    E_signal_f = np.fft.fft(E_signal)
    E_reference_f = np.fft.fft(E_reference)
    f = np.fft.fftfreq(E_signal.shape[0], dt)
    bandpass = np.exp( -((f-filter_f0)**filter_order)/(2*filter_sigma**filter_order))
    impulse =  np.fft.ifft(bandpass*np.nan_to_num(E_signal_f/E_reference_f))
    return impulse

def shift_phase_amplitude(Et, dt: float, time_shift: float, phase_shift: float, amplitude_factor: float):
    """
    Apply a time shift, phase shift, and amplitude factor to a field

    Args:
        Et: the waveform to be modified
        dt: the time step of the waveform
        time_shift (float): the time delay to apply
        phase_shift (float): the phase shift to apply (radians)
        amplitude_factor (float): the amplitude factor to apply (unitless)

    Returns:
        np.ndarray: the modified waveform
    """
    Et_f = np.fft.fft(Et)
    f = np.fft.fftfreq(Et.shape[0],dt)
    return np.fft.ifft(amplitude_factor * np.exp(1j*(phase_shift - (time_shift*2*np.pi)*f)) * Et_f)

def minimize_response_difference(response, reference):
    """
    Shift the delay, phase, and amplitude of the reference field to minimize the squared-intensity-envelope
    after subtracting them.

    Args:
        response: The signal waveform
        reference: The reference waveform

    Returns:
        np.ndarray: The modified reference field
    """
    # Contributed by Nick Karpowicz
    #construct initial guess of the time, phase, and amplitude offsets
    peak_location_response = np.argmax(np.abs(response))
    peak_location_reference = np.argmax(np.abs(reference))
    start_phase_shift = np.angle(response[peak_location_response]) - np.angle(reference[peak_location_reference])
    start_time_shift = peak_location_response - peak_location_reference
    start_amplitude = np.abs(reference[peak_location_reference])/np.abs(response[peak_location_response])
    start_values = [start_time_shift,start_phase_shift,start_amplitude]

    #function to get the envelope (to 4th power) of the difference between reference and response
    def get_residual(parameters):
        adjusted_reference = shift_phase_amplitude(reference,1.0, parameters[0], parameters[1], parameters[2])
        return np.abs(response-adjusted_reference)**4

    #minimize residuals
    res = opt.least_squares(get_residual, start_values, ftol = 1e-12, max_nfev=16384)
    return shift_phase_amplitude(reference,1.0, res.x[0], res.x[1], res.x[2])

def get_effective_response(E_signal, E_reference, dt: float, filter_f0: float, filter_sigma: float, filter_order: int):
    """
    Use the above three functions to calculate the effective impulse response with the reactive response removed

    Args:
        E_signal: the signal waveform array
        E_reference: the reference waveform
        dt (float): the time step
        filter_f0 (float): central frequency of the bandpass
        filter_sigma (float): width of the bandpass
        filter_order (int): order of the bandpass

    Returns:
        np.ndarray: the impuse response
    """
    # Contributed by Nick Karpowicz
    analytic_signal = sig.hilbert(np.real(E_signal))
    analytic_reference = sig.hilbert(np.real(E_reference))

    response = filtered_impulse_response(analytic_signal, analytic_reference, dt, filter_f0, filter_sigma, filter_order)
    window_response = filtered_impulse_response(analytic_reference, analytic_reference, dt, filter_f0, filter_sigma, filter_order)
    matched_window_response = minimize_response_difference(response, window_response)
    return np.real(response-matched_window_response)
