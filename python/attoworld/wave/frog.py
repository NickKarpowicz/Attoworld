"""Frog reconstruction and handling."""

import numpy as np

from ..attoworld_rs import FrogType, generate_spectrogram_with_error, rust_frog
from ..data import ComplexEnvelope, FrogData, IntensitySpectrum, Spectrogram
from ..numeric import find_maximum_location, interpolate


# Helper functions
def shift_to_zero_and_normalize(Et):
    """Fix the fact that the reconstructed pulse from FROG has random delay."""
    max_loc, max_val = find_maximum_location(np.abs(Et))
    Ew = np.fft.fft(Et)
    _f = np.fft.fftfreq(len(Et))
    return np.fft.ifft(
        np.exp(1j * 2 * np.pi * (max_loc + len(Et) / 2) * _f) * Ew / max_val
    )


def bundle_frog_reconstruction(
    t,
    result,
    measurement,
    frog_type: FrogType,
    f0: float = 375e12,
    interpolation_factor: int = 100,
    gate=None,
):
    """Turn the results of a FROG reconstruction into a FrogData struct.

    Args:
        t: the time vector (s)
        result: the reconstructed complex envelope
        measurement: the measured spectrogram, sqrt-ed and fftshift-ed
        frog_type (FrogType): the type of frog used
        f0 (float): the central frequency (Hz)
        interpolation_factor (int): factor by which to interpolate to get a valid waveform (Nyquist)
        gate: the gate complex envelope (optional, used in xfrog)

    Returns:
        FrogData: the bundled data

    """
    f = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    sg_freq = np.fft.fftshift(f) + 2 * f0

    measured_not_sqrt = measurement**2
    (result_sg_data, g_prime_error, g_error) = generate_spectrogram_with_error(
        frog_type, result, gate, measured_not_sqrt / np.linalg.norm(measured_not_sqrt)
    )

    if gate is None:
        gate = result

    result_sg = Spectrogram(
        data=np.fft.fftshift(result_sg_data, axes=0),
        time=t,
        freq=sg_freq,
    )
    measurement_sg = Spectrogram(
        data=np.fft.fftshift(measured_not_sqrt, axes=0), time=t, freq=sg_freq
    )
    result_ce = ComplexEnvelope(
        time=t, dt=(t[1] - t[0]), carrier_frequency=f0, envelope=result
    ).to_waveform(interpolation_factor=interpolation_factor)
    result_cs = result_ce.to_complex_spectrum()

    return FrogData(
        measured_spectrogram=measurement_sg,
        pulse=result_ce,
        reconstructed_spectrogram=result_sg,
        spectrum=result_cs,
        raw_reconstruction=result,
        f0=f0,
        dt=(t[1] - t[0]),
        g_prime_error=g_prime_error,
        g_error=g_error,
    )


def generate_gate_from_frog(
    reconstructed_gate: FrogData, target_spectrogram: Spectrogram
):
    """Generate a gate array to be used in an XFROG reconstruction.

    Args:
        reconstructed_gate: a previous FROG result of the gate pulse
        target_spectrogram: the XFROG spectrogram to match the gate to

    Returns:
        np.ndarray: the gate to give to the reconstruct_xfrog_core function

    """
    t_gate = reconstructed_gate.dt * np.arange(
        reconstructed_gate.raw_reconstruction.shape[0]
    )
    t_gate -= np.mean(t_gate)
    t_reconstruction = target_spectrogram.time - np.mean(target_spectrogram.time)
    interpolated_real = interpolate(
        t_reconstruction,
        t_gate,
        np.array(np.real(reconstructed_gate.raw_reconstruction)),
        inputs_are_sorted=True,
    )
    interpolated_imag = interpolate(
        t_reconstruction,
        t_gate,
        np.array(np.imag(reconstructed_gate.raw_reconstruction)),
        inputs_are_sorted=True,
    )
    return interpolated_real + 1j * interpolated_imag


def fix_aliasing(result):
    """Check if the reconstruction is aliased.

    Args:
        result: the result to check

    """
    offset = int(len(result) / 2)
    firstprod = np.real(result[offset]) * np.real(result[offset + 1])
    if firstprod < 0.0:
        return np.fft.ifft(np.fft.fftshift(np.fft.fft(result)))
    return result


def reconstruct_frog(
    measurement: Spectrogram,
    test_iterations: int = 100,
    polish_iterations=5000,
    repeats: int = 256,
    frog_type: FrogType = FrogType.Shg,
    spectrum: IntensitySpectrum | None = None,
    xfrog_gate: FrogData | None = None,
    roi=None,
    ptychographic_threshhold: float | None = None,
):
    """Run the core FROG loop several times and pick the best result.

    Args:
        measurement (np.ndarray): measured spectrogram, sqrt + fftshift(axes=0)
        test_iterations (int): number of iterations for the multiple tests
        polish_iterations (int): number of extra iterations to apply to the winner
        repeats (int): number of different initial guesses to try
        frog_type (FrogType): type of nonlinear effect. Options: SHG, THG, and Kerr
        spectrum (IntensitySpectrum): optional spectrum to use to constrain the spectrum of the retrieved pulse
        xfrog_gate (FrogData): gate to use for xfrog
        roi (np.ndarray): array of boolean values, with the same length as the frequency vector, saying if each point is within the region of interested for ptychographic FROG
        ptychographic_threshhold (float): the threshold (gamma) to use in denoising the ptychographic frog
    Returns:
    FrogData: the completed reconstruction

    """
    sqrt_sg = np.fft.fftshift(
        np.sqrt(measurement.data - np.min(measurement.data[:])), axes=0
    )
    sqrt_sg /= np.max(sqrt_sg)
    measurement_norm = sqrt_sg**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    measured_gate = None
    match frog_type:
        case FrogType.Shg:
            f0 = float(np.mean(measurement.freq) / 2.0)
        case FrogType.PtychographicShg:
            f0 = float(np.mean(measurement.freq) / 2.0)
        case FrogType.Thg:
            f0 = float(np.mean(measurement.freq) / 3.0)
        case FrogType.Xfrog:
            if xfrog_gate is not None:
                f0 = float(np.mean(measurement.freq) - xfrog_gate.f0)
                measured_gate = generate_gate_from_frog(xfrog_gate, measurement)
            else:
                raise ValueError(
                    "Must provide a measured gate pulse for XFROG, using the xfrog_gate input parameter."
                )
        case FrogType.Kerr | _:
            f0 = float(np.mean(measurement.freq))

    if spectrum is not None:
        spec_freq, spec = spectrum.get_frequency_spectrum()
        spec -= np.min(spec)
        spectral_constraint = np.sqrt(
            interpolate(
                measurement.freq - np.mean(measurement.freq) + f0,
                spec_freq,
                spec,
                inputs_are_sorted=False,
            )
        )
        spectral_constraint = np.array(np.fft.fftshift(spectral_constraint))

        # Correct marginal for SHG-FROG if a spectral constraint is applied
        if frog_type == FrogType.Shg or frog_type == FrogType.PtychographicShg:
            spectrogram_marginal = np.sum(sqrt_sg**2, axis=1)
            spectrum_autocorrelation = np.abs(
                np.fft.ifft(np.fft.fft(spectral_constraint) ** 2)
            )
            wiener_factor = np.where(
                spectrogram_marginal > 0.0,
                (1.0 / spectrogram_marginal)
                * (1.0 / (1.0 + 1.0 / (spectrogram_marginal**2 * 1000.0))),
                0.0,
            )
            amp_factor = np.where(
                spectrogram_marginal > 0.0,
                wiener_factor * spectrum_autocorrelation,
                0.0,
            )
            sqrt_sg = np.array(
                np.sqrt(amp_factor[:, np.newaxis] * sqrt_sg**2), dtype=float
            )

    else:
        spectral_constraint = None

    (pulse_out, gate_out, g_error, best_trial_index, best_finishing_index) = rust_frog(
        np.array(sqrt_sg),
        None,
        trial_pulses=repeats,
        iterations=test_iterations,
        finishing_iterations=polish_iterations,
        frog_type=frog_type,
        spectrum=spectral_constraint,
        measured_gate=measured_gate,
        roi=roi,
        ptycho_threshhold=ptychographic_threshhold,
    )
    nyquist_factor = int(
        np.ceil(2 * (measurement.time[1] - measurement.time[0]) * measurement.freq[-1])
    )

    if (frog_type != FrogType.Xfrog) and (frog_type != FrogType.Blindfrog):
        pulse_out = shift_to_zero_and_normalize(pulse_out)
        gate_out = shift_to_zero_and_normalize(gate_out)

    result = bundle_frog_reconstruction(
        frog_type=frog_type,
        t=measurement.time,
        result=pulse_out,
        measurement=sqrt_sg,
        f0=f0,
        interpolation_factor=nyquist_factor,
        gate=gate_out,
    )

    gate_result = bundle_frog_reconstruction(
        frog_type=frog_type,
        t=measurement.time,
        result=gate_out,
        measurement=sqrt_sg,
        f0=f0,
        interpolation_factor=nyquist_factor,
        gate=gate_out,
    )

    return result, gate_result, best_trial_index, best_finishing_index
