"""For processing Amelie's data."""

import numpy as np
import scipy.optimize as opt
import scipy.signal as sig


def mean_offset_tukey(signal):
    """Apply DC offset removal and a Tukey window to a waveform.

    Args:
        signal: the array describing the measured waveform

    Returns:
        np.ndarray: the modified signal

    """
    mean_signal = np.mean(signal, axis=0)
    mean_signal -= np.mean(mean_signal)
    mean_signal *= sig.windows.tukey(mean_signal.shape[0])
    return mean_signal


def filtered_impulse_response(
    E_signal,
    E_reference,
    dt: float,
    filter_f0: float,
    filter_sigma: float,
    filter_order: int,
):
    """Calculate the filtered impulse response given by a signal and reference pair, with applied bandpass filter.

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
    E_signal_f = np.fft.fft(E_signal)
    E_reference_f = np.fft.fft(E_reference)
    f = np.fft.fftfreq(E_signal.shape[0], dt)
    bandpass = np.exp(
        -((f - filter_f0) ** filter_order) / (2 * filter_sigma**filter_order)
    )
    return np.fft.ifft(bandpass * np.nan_to_num(E_signal_f / E_reference_f))


def shift_phase_amplitude(
    Et, dt: float, time_shift: float, phase_shift: float, amplitude_factor: float
):
    """Apply a time shift, phase shift, and amplitude factor to a field.

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
    f = np.fft.fftfreq(Et.shape[0], dt)
    return np.fft.ifft(
        amplitude_factor
        * np.exp(1j * (phase_shift - (time_shift * 2 * np.pi) * f))
        * Et_f
    )


def minimize_response_difference(response, reference):
    """Shift the delay, phase, and amplitude of the reference field to minimize the squared-intensity-envelope
    after subtracting them.

    Args:
        response: The signal waveform
        reference: The reference waveform

    Returns:
        np.ndarray: The modified reference field

    """
    # construct initial guess of the time, phase, and amplitude offsets
    peak_location_response = np.argmax(np.abs(response))
    peak_location_reference = np.argmax(np.abs(reference))
    start_phase_shift = np.angle(response[peak_location_response]) - np.angle(
        reference[peak_location_reference]
    )
    start_time_shift = peak_location_response - peak_location_reference
    start_amplitude = np.abs(reference[peak_location_reference]) / np.abs(
        response[peak_location_response]
    )
    start_values = [start_time_shift, start_phase_shift, start_amplitude]

    # function to get the envelope (to 4th power) of the difference between reference and response
    def get_residual(parameters):
        adjusted_reference = shift_phase_amplitude(
            reference, 1.0, parameters[0], parameters[1], parameters[2]
        )
        return np.abs(response - adjusted_reference) ** 4

    # minimize residuals
    res = opt.least_squares(get_residual, start_values, ftol=1e-12, max_nfev=16384)
    return shift_phase_amplitude(reference, 1.0, res.x[0], res.x[1], res.x[2])


def get_effective_response(
    E_signal,
    E_reference,
    dt: float,
    filter_f0: float,
    filter_sigma: float,
    filter_order: int,
):
    """Calculate the effective impulse response with the reactive response removed.

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
    analytic_signal = sig.hilbert(np.real(E_signal))
    analytic_reference = sig.hilbert(np.real(E_reference))

    response = filtered_impulse_response(
        analytic_signal, analytic_reference, dt, filter_f0, filter_sigma, filter_order
    )
    window_response = filtered_impulse_response(
        analytic_reference,
        analytic_reference,
        dt,
        filter_f0,
        filter_sigma,
        filter_order,
    )
    matched_window_response = minimize_response_difference(response, window_response)
    return np.real(response - matched_window_response)
