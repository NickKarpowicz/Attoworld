import numpy as np
from ..data import ComplexEnvelope, Spectrogram, FrogData
from ..numeric import find_maximum_location


# Helper functions
def shift_to_zero_and_normalize(Et):
    """
    Fix the fact that the reconstructed pulse from FROG has random delay
    """
    max_loc, max_val = find_maximum_location(np.abs(Et))
    Ew = np.fft.fft(Et)
    _f = np.fft.fftfreq(len(Et))
    return np.fft.ifft(
        np.exp(1j * 2 * np.pi * (max_loc + len(Et) / 2) * _f) * Ew / max_val
    )


def bundle_frog_reconstruction(
    t, result, measurement, f0: float = 375e12, interpolation_factor: int = 100
):
    """
    Turn the results of a FROG reconstruction into a FrogData struct

    Args:
        t: the time vector (s)
        result: the reconstructed complex envelope
        measurement: the measured spectrogram, sqrt-ed and fftshift-ed
        f0 (float): the central frequency (Hz)
        interpolation_factor (int): factor by which to interpolate to get a valid waveform (Nyquist)

    Returns:
        FrogData: the bundled data
    """
    f = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    sg_freq = np.fft.fftshift(f) + 2 * f0
    result_sg = Spectrogram(
        data=np.fft.fftshift(
            np.abs(generate_shg_spectrogram(result, result)) ** 2, axes=0
        ),
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
def generate_shg_spectrogram(Et, Gt):
    """
    Generate a SHG spectrogram, same pattern as in FROG book

    Args:
        Et: field
        Gt: gate (same as field unless XFROG)
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
    """np.roll, but pulse entering from other side set to zero"""
    rolled = np.roll(data, step)
    if step > 0:
        rolled[0:step] = 0.0
    elif step < 0:
        rolled[(-1 + step) : :] = 0.0
    return rolled


def apply_iteration(Et, Gt, meas_sqrt):
    """
    Apply an iteration of the generalized projections SHG-FROG

    Args:
        Et: field
        Gt: gate
        meas_sqrt: the measurement, sqrt-ed and fftshift, axes=0

    Returns:
        np.ndarray, np.ndarray: field, gate"""
    new_sg = generate_shg_spectrogram(Et, Gt)
    new_sg = meas_sqrt * np.exp(1j * np.angle(new_sg))
    new_sg = np.fft.ifft(new_sg, axis=0)
    for _i in range(len(Et)):
        new_sg[_i, :] = blank_roll(new_sg[_i, :], _i - +int(Et.shape[0] / 2))
    u, s, v = np.linalg.svd(new_sg)
    field = u[:, 0].squeeze()
    gate = v[0, :].squeeze()
    return field, gate


def calculate_g_error(measurement, pulse):
    """Calculate G' error helper function"""
    meas_squared = measurement**2
    meas_squared /= np.linalg.norm(meas_squared)
    reconstructed = np.abs(generate_shg_spectrogram(pulse, pulse)) ** 2
    reconstructed /= np.linalg.norm(reconstructed)
    return np.sqrt(
        np.sum((meas_squared[:] - reconstructed[:]) ** 2) / np.sum(meas_squared[:] ** 2)
    )


def reconstruct_shg_frog_core(
    measurement_sg_sqrt, guess=None, max_iterations: int = 200
):
    """
    Run the core FROG loop

    Args:
        measurement: measured spectrogram, sqrt + fftshift(axes=0)
        guess: initial guess for the field (will be randomly generated if not set)
        max_iterations: number of iterations to run

    Returns:
        np.ndarray: the reconstructed field
    """
    if guess is None:
        guess_pulse = np.random.randn(
            measurement_sg_sqrt.shape[0]
        ) + 1j * np.random.randn(measurement_sg_sqrt.shape[0])
        guess, gate = apply_iteration(guess_pulse, guess_pulse, measurement_sg_sqrt)
    else:
        gate = guess
    best = shift_to_zero_and_normalize(guess + gate)
    current = best
    best_error = calculate_g_error(measurement_sg_sqrt, best)
    for _i in range(max_iterations):
        guess, gate = apply_iteration(current, current, measurement_sg_sqrt)
        current = shift_to_zero_and_normalize(guess + gate)
        current = fix_aliasing(current)
        current_error = calculate_g_error(measurement_sg_sqrt, current)
        if current_error < best_error:
            best_error = current_error
            best = current
    return best


def fix_aliasing(result):
    offset = int(len(result) / 2)
    firstprod = np.real(result[offset]) * np.real(result[offset + 1])
    if firstprod < 0.0:
        return np.fft.ifft(np.fft.fftshift(np.fft.fft(result)))
    else:
        return result


def reconstruct_shg_frog(
    measurement: Spectrogram,
    test_iterations: int = 100,
    polish_iterations=5000,
    repeats: int = 256,
):
    """
    Run the core FROG loop several times and pick the best result

    Args:
        measurement (np.ndarray): measured spectrogram, sqrt + fftshift(axes=0)
        test_iterations (int): number of iterations for the multiple tests
        polish_iteration (int): number of extra iterations to apply to the winner
        repeats (int): number of different initial guesses to try

    Returns:
    FrogData: the completed reconstruction
    """
    sqrt_sg = np.fft.fftshift(
        np.sqrt(measurement.data - np.min(measurement.data[:])), axes=0
    )
    sqrt_sg /= np.max(sqrt_sg)
    results = np.zeros((sqrt_sg.shape[0], repeats), dtype=np.complex128)
    errors = np.zeros(repeats, dtype=float)
    for _i in range(repeats):
        results[:, _i] = reconstruct_shg_frog_core(
            sqrt_sg, max_iterations=test_iterations
        )
        errors[_i] = calculate_g_error(sqrt_sg, results[:, _i])
    min_error_index = np.argmin(errors)
    result = reconstruct_shg_frog_core(
        sqrt_sg, guess=results[:, min_error_index], max_iterations=polish_iterations
    )
    return bundle_frog_reconstruction(
        t=measurement.time,
        result=result,
        measurement=sqrt_sg,
        f0=float(np.mean(measurement.freq) / 2.0),
    )
