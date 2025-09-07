"""Frog reconstruction and handling."""

import numpy as np

from ..attoworld_rs import FrogType, frog_iteration, rust_frog
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
    f0: float = 375e12,
    interpolation_factor: int = 100,
    gate=None,
):
    """Turn the results of a FROG reconstruction into a FrogData struct.

    Args:
        t: the time vector (s)
        result: the reconstructed complex envelope
        measurement: the measured spectrogram, sqrt-ed and fftshift-ed
        f0 (float): the central frequency (Hz)
        interpolation_factor (int): factor by which to interpolate to get a valid waveform (Nyquist)
        gate: the gate complex envelope (optional, used in xfrog)

    Returns:
        FrogData: the bundled data

    """
    if gate is None:
        gate = result
    f = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    sg_freq = np.fft.fftshift(f) + 2 * f0
    result_sg = Spectrogram(
        data=np.fft.fftshift(np.abs(generate_spectrogram(result, gate)) ** 2, axes=0),
        time=t,
        freq=sg_freq,
    )
    measurement_sg = Spectrogram(
        data=np.fft.fftshift(np.abs(measurement) ** 2, axes=0), time=t, freq=sg_freq
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
    )


# FROG functions
def generate_spectrogram(Et, Gt):
    """Generate a spectrogram, same pattern as in FROG book.

    Args:
        Et: field
        Gt: gate

    Returns:
        np.ndarray: the complex spectrogram

    """
    spectrogram_timetime = np.outer(Et, Gt)
    for _i in range(Et.shape[0]):
        spectrogram_timetime[_i, :] = blank_roll(
            spectrogram_timetime[_i, :], -_i + int(Et.shape[0] / 2)
        )

    return np.fft.fft(spectrogram_timetime, axis=0)

def blank_roll(data: np.ndarray, step):
    """np.roll, but pulse entering from other side set to zero."""
    rolled = np.roll(data, step)
    if step > 0:
        rolled[:step] = 0.0
    elif step < 0:
        rolled[step:] = 0.0
    return rolled


def apply_iteration(Et, Gt, meas_sqrt):
    """Apply an iteration of the generalized projections SHG-FROG.

    Args:
        Et: field
        Gt: gate
        meas_sqrt: the measurement, sqrt-ed and fftshift, axes=0

    Returns:
        np.ndarray, np.ndarray: field, gate

    """
    new_sg = generate_spectrogram(Et, Gt)
    new_sg = meas_sqrt * np.exp(1j * np.angle(new_sg))
    new_sg = np.fft.ifft(new_sg, axis=0)

    field = np.mean(new_sg, axis=1)
    for _i in range(len(Et)):
        new_sg[_i, :] = blank_roll(new_sg[_i, :], _i - int(Et.shape[0] / 2))

    # Principle component based pulse extraction
    # u, s, v = np.linalg.svd(new_sg)
    # field = u[:, 0].squeeze()
    # gate = v[0, :].squeeze()

    # Simpler extraction

    gate = np.mean(new_sg, axis=0)

    return field, gate


def calculate_g_error(measurement_normalized, pulse, gate=None):
    """Calculate G' error helper function."""
    if gate is None:
        recon_normalized = np.abs(generate_spectrogram(pulse, pulse)) ** 2
    else:
        recon_normalized = np.abs(generate_spectrogram(pulse, gate)) ** 2
    recon_normalized /= np.linalg.norm(recon_normalized)
    return np.sqrt(
        np.sum((measurement_normalized[:] - recon_normalized[:]) ** 2)
        / np.sum(measurement_normalized[:] ** 2)
    )


def apply_spectral_constraint(field, spectrum):
    """Apply a set of spectral amplitudes to a field, keeping only the phase."""
    if spectrum is not None:
        return np.fft.ifft(spectrum * np.exp(1j * np.angle(np.fft.fft(field))))
    return field


def guess_from_pulse_and_gate(pulse, gate, nonlinearity: str = "SHG", spectrum=None):
    """Use the retrieved gate and pulse to generate a new guess of the pulse."""
    match nonlinearity:
        case "SHG":
            return shift_to_zero_and_normalize(
                apply_spectral_constraint(pulse + gate, spectrum)
            )
        case "THG":
            return shift_to_zero_and_normalize(
                apply_spectral_constraint(pulse + np.conj(pulse) * gate, spectrum)
            )
        case "Kerr":
            return shift_to_zero_and_normalize(
                apply_spectral_constraint(pulse, spectrum)
            )


def gate_from_pulse(pulse, nonlinearity: str = "SHG"):
    """Generate the gate using the current retrieved pulse."""
    match nonlinearity:
        case "SHG":
            return pulse
        case "THG":
            return pulse * pulse
        case "Kerr":
            return np.conj(pulse) * pulse


def reconstruct_frog_core(
    measurement_sg_sqrt,
    guess=None,
    max_iterations: int = 200,
    nonlinearity: str = "SHG",
    spectrum=None,
):
    """Run the core FROG loop.

    Args:
        measurement_sg_sqrt: measured spectrogram, sqrt + fftshift(axes=0)
        guess: initial guess for the field (will be randomly generated if not set)
        max_iterations: number of iterations to run
        nonlinearity (str): nonlinear interaction that made the FROG. Choices are SHG, THG, and Kerr
        spectrum: optional spectral intensity constraint
    Returns:
        np.ndarray: the reconstructed field

    """
    measurement_norm = measurement_sg_sqrt**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    if guess is None:
        guess = np.random.randn(measurement_sg_sqrt.shape[0]) + 1j * np.random.randn(
            measurement_sg_sqrt.shape[0]
        )
        guess = apply_spectral_constraint(guess, spectrum)
    else:
        gate = guess
    gate = gate_from_pulse(guess, nonlinearity)
    best = guess
    best_error = calculate_g_error(measurement_norm, best)
    for _i in range(max_iterations):
        guess, gate = frog_iteration(
            np.array(guess), np.array(gate), np.array(measurement_sg_sqrt)
        )
        guess = guess_from_pulse_and_gate(guess, gate, nonlinearity, spectrum)
        guess = fix_aliasing(guess)
        gate = gate_from_pulse(guess, nonlinearity)
        current_error = calculate_g_error(measurement_norm, guess)
        if current_error < best_error:
            best_error = current_error
            best = guess
    return best


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


def reconstruct_xfrog_core(
    measurement_sg_sqrt, gate, guess=None, max_iterations: int = 200
):
    """Run the core FROG loop.

    Args:
        measurement_sg_sqrt: measured spectrogram, sqrt + fftshift(axes=0)
        gate: complex-valued time-domain gate pulse
        guess: initial guess for the field (will be randomly generated if not set)
        max_iterations: number of iterations to run

    Returns:
        np.ndarray: the reconstructed field

    """
    measurement_norm = measurement_sg_sqrt**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    if guess is None:
        guess = np.random.randn(measurement_sg_sqrt.shape[0]) + 1j * np.random.randn(
            measurement_sg_sqrt.shape[0]
        )
    guess = shift_to_zero_and_normalize(guess)
    best = np.array(guess)
    best_error = calculate_g_error(measurement_norm, best, gate)
    for _i in range(max_iterations):
        guess, _ = apply_iteration(guess, np.array(gate), measurement_sg_sqrt)
        # guess = shift_to_zero_and_normalize(guess)
        # guess = fix_aliasing(guess)
        current_error = calculate_g_error(measurement_norm, guess, gate)
        if current_error < best_error:
            best_error = current_error
            best = guess
    return best


def reconstruct_blindfrog_core(
    measurement_sg_sqrt, gate=None, guess=None, max_iterations: int = 200
):
    """Run the core FROG loop.

    Args:
        measurement_sg_sqrt: measured spectrogram, sqrt + fftshift(axes=0)
        gate: complex-valued time-domain gate pulse (will be randomly generated if not set)
        guess: initial guess for the field (will be randomly generated if not set)
        max_iterations: number of iterations to run

    Returns:
        np.ndarray: the reconstructed field

    """
    measurement_norm = measurement_sg_sqrt**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    if guess is None:
        guess = np.random.randn(measurement_sg_sqrt.shape[0]) + 1j * np.random.randn(
            measurement_sg_sqrt.shape[0]
        )
    if gate is None:
        gate = np.random.randn(measurement_sg_sqrt.shape[0]) + 1j * np.random.randn(
            measurement_sg_sqrt.shape[0]
        )
    best_pulse = guess
    best_gate = gate
    best_error = calculate_g_error(measurement_norm, guess, gate)
    for _i in range(max_iterations):
        guess, gate = apply_iteration(guess, gate, measurement_sg_sqrt)
        current_error = calculate_g_error(measurement_norm, guess, gate)
        if current_error < best_error:
            best_error = current_error
            best_pulse = guess
            best_gate = gate
    return best_pulse, best_gate


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
    xfrog_gate: FrogData | None = None
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
        case FrogType.Thg:
            f0 = float(np.mean(measurement.freq) / 3.0)
        case FrogType.Xfrog:
            if xfrog_gate is not None:
                f0 = float(np.mean(measurement.freq) - xfrog_gate.f0)
                measured_gate = generate_gate_from_frog(xfrog_gate, measurement)
            else:
                raise ValueError("Must provide a measured gate pulse for XFROG, using the xfrog_gate input parameter.")
        case FrogType.Kerr | _:
            f0 = float(np.mean(measurement.freq))

    if spectrum is not None:
        spec_freq, spec = spectrum.get_frequency_spectrum()
        spectrum = np.sqrt(
            interpolate(
                measurement.freq - np.mean(measurement.freq) + f0,
                spec_freq,
                spec,
                inputs_are_sorted=False,
            )
        )

    (pulse_out, gate_out, g_error) = rust_frog(
        np.array(sqrt_sg),
        None,
        trial_pulses=repeats,
        iterations=test_iterations,
        finishing_iterations=polish_iterations,
        frog_type=frog_type,
        spectrum=spectrum,
        measured_gate=measured_gate,
    )
    nyquist_factor = int(
        np.ceil(2 * (measurement.time[1] - measurement.time[0]) * measurement.freq[-1])
    )

    pulse_out = shift_to_zero_and_normalize(pulse_out)
    gate_out = shift_to_zero_and_normalize(gate_out)

    result = bundle_frog_reconstruction(
        t=measurement.time,
        result=pulse_out,
        measurement=sqrt_sg,
        f0=f0,
        interpolation_factor=nyquist_factor,
        gate = gate_out
    )

    gate_result = bundle_frog_reconstruction(
        t=measurement.time,
        result=gate_out,
        measurement=sqrt_sg,
        f0=f0,
        interpolation_factor=nyquist_factor,
        gate = gate_out
    )

    return result, gate_result


def reconstruct_xfrog(
    measurement: Spectrogram,
    gate: FrogData,
    test_iterations: int = 100,
    polish_iterations=5000,
    repeats: int = 256,
):
    """Run the core FROG loop several times and pick the best result.

    Args:
        measurement (np.ndarray): measured spectrogram, sqrt + fftshift(axes=0)
        gate (FrogData): the reconstructed gate
        test_iterations (int): number of iterations for the multiple tests
        polish_iterations (int): number of extra iterations to apply to the winner
        repeats (int): number of different initial guesses to try

    Returns:
    FrogData: the completed reconstruction

    """
    gate_pulse = generate_gate_from_frog(gate, measurement)
    sqrt_sg = np.fft.fftshift(
        np.sqrt(measurement.data - np.min(measurement.data[:])), axes=0
    )
    sqrt_sg /= np.max(sqrt_sg)
    measurement_norm = sqrt_sg**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    results = np.zeros((sqrt_sg.shape[0], repeats), dtype=np.complex128)
    errors = np.zeros(repeats, dtype=float)
    for _i in range(repeats):
        results[:, _i] = reconstruct_xfrog_core(
            sqrt_sg, gate_pulse, max_iterations=test_iterations
        )
        errors[_i] = calculate_g_error(measurement_norm, results[:, _i], gate_pulse)
    min_error_index = np.argmin(errors)
    result = reconstruct_xfrog_core(
        sqrt_sg,
        gate_pulse,
        guess=results[:, min_error_index],
        max_iterations=polish_iterations,
    )
    nyquist_factor = int(
        np.ceil(2 * (measurement.time[1] - measurement.time[0]) * measurement.freq[-1])
    )
    return bundle_frog_reconstruction(
        t=measurement.time,
        result=result,
        measurement=sqrt_sg,
        f0=float(np.mean(measurement.freq) - gate.f0),
        gate=gate_pulse,
        interpolation_factor=nyquist_factor,
    ), gate_pulse


def reconstruct_blindfrog(
    measurement: Spectrogram,
    test_iterations: int = 100,
    polish_iterations=5000,
    repeats: int = 256,
):
    """Run the core FROG loop several times and pick the best result.

    Args:
        measurement (np.ndarray): measured spectrogram, sqrt + fftshift(axes=0)
        gate (FrogData): the reconstructed gate
        test_iterations (int): number of iterations for the multiple tests
        polish_iterations (int): number of extra iterations to apply to the winner
        repeats (int): number of different initial guesses to try

    Returns:
    FrogData: the completed reconstruction

    """
    sqrt_sg = np.fft.fftshift(
        np.sqrt(measurement.data - np.min(measurement.data[:])), axes=0
    )
    sqrt_sg /= np.max(sqrt_sg)
    measurement_norm = sqrt_sg**2
    measurement_norm = measurement_norm / np.linalg.norm(measurement_norm)
    results = np.zeros((sqrt_sg.shape[0], repeats), dtype=np.complex128)
    gates = np.zeros((sqrt_sg.shape[0], repeats), dtype=np.complex128)
    errors = np.zeros(repeats, dtype=float)
    for _i in range(repeats):
        results[:, _i], gates[:, _i] = reconstruct_blindfrog_core(
            sqrt_sg, max_iterations=test_iterations
        )
        errors[_i] = calculate_g_error(measurement_norm, results[:, _i], gates[:, _i])
    min_error_index = np.argmin(errors)
    result, gate = reconstruct_blindfrog_core(
        sqrt_sg,
        gate=gates[:, min_error_index],
        guess=results[:, min_error_index],
        max_iterations=polish_iterations,
    )
    nyquist_factor = int(
        np.ceil(2 * (measurement.time[1] - measurement.time[0]) * measurement.freq[-1])
    )
    result_A = bundle_frog_reconstruction(
        t=measurement.time,
        result=result,
        measurement=sqrt_sg,
        f0=float(np.mean(measurement.freq) / 2),
        gate=gate,
        interpolation_factor=nyquist_factor,
    )
    result_B = bundle_frog_reconstruction(
        t=measurement.time,
        result=gate,
        measurement=sqrt_sg,
        f0=float(np.mean(measurement.freq) / 2),
        gate=result,
        interpolation_factor=nyquist_factor,
    )
    return result_A, result_B
