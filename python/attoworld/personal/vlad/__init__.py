# -*- coding: utf-8 -*-
# Author: Vladislav S. Yakovlev
"""A collection of utility functions and a class for physics calculations.

This module provides various numerical helper functions, including FFT
utilities, windowing functions, and data analysis tools, along with a
class defining atomic unit constants.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.linalg  # Explicitly import submodule used

# Type alias for 1D or 2D NumPy arrays primarily holding float data.
# Using Any for dtype as functions sometimes handle complex numbers.
ArrayLike = np.ndarray[Any, Any]


def nextpow2(number: float) -> int:
    """Computes the exponent for the smallest power of 2 >= number.

    Args:
      number: The input number.

    Returns:
      The smallest integer exponent `exp` such that 2**exp >= number.

    """
    return int(np.ceil(np.log2(number)))


def soft_window(x_grid: ArrayLike, x_begin: float, x_end: float) -> ArrayLike:
    """Computes a soft window function.

    The window smoothly transitions from 1 to 0 over the interval
    [min(x_begin, x_end), max(x_begin, x_end)].
    If x_begin <= x_end, it's a cosine-squared fall-off.
    If x_begin > x_end, it's a sine-squared rise-on.

    Args:
      x_grid: A 1D array of x-coordinates.
      x_begin: The x-coordinate where the transition starts (value is 1).
      x_end: The x-coordinate where the transition ends (value is 0).

    Returns:
      A 1D array of the same size as x_grid, containing the window values.

    """
    window = np.zeros_like(x_grid)
    x_min_transition = min(x_begin, x_end)
    x_max_transition = max(x_begin, x_end)

    # Determine indices for different parts of the window
    indices_before_transition = np.where(x_grid < x_min_transition)[0]
    idx_transition_start = (
        indices_before_transition[-1] + 1 if indices_before_transition.size > 0 else 0
    )

    indices_after_transition = np.where(x_grid > x_max_transition)[0]
    idx_transition_end = (
        indices_after_transition[0]
        if indices_after_transition.size > 0
        else len(x_grid)
    )

    # Define the transition region
    x_transition = x_grid[idx_transition_start:idx_transition_end]

    if x_begin <= x_end:  # Window goes from 1 down to 0
        window[:idx_transition_start] = 1.0
        if (
            idx_transition_end > idx_transition_start
            and x_max_transition > x_min_transition
        ):
            window[idx_transition_start:idx_transition_end] = (
                np.cos(
                    np.pi
                    / 2.0
                    * (x_transition - x_min_transition)
                    / (x_max_transition - x_min_transition)
                )
                ** 2
            )
        # Values after x_end remain 0 (initialized)
    else:  # Window goes from 0 up to 1 (x_begin > x_end, so x_min_transition = x_end)
        window[idx_transition_end:] = 1.0
        if (
            idx_transition_end > idx_transition_start
            and x_max_transition > x_min_transition
        ):
            window[idx_transition_start:idx_transition_end] = (
                np.sin(
                    np.pi
                    / 2.0
                    * (x_transition - x_min_transition)
                    / (x_max_transition - x_min_transition)
                )
                ** 2
            )
        # Values before x_begin (which is x_max_transition) remain 0 (initialized)
    return window


def get_significant_part_indices_v1(
    array_data: ArrayLike, threshold: float = 1e-8
) -> Tuple[int, int]:
    """Returns indices (i1, i2) for the slice A[i1:i2] containing the
    significant part of the array.

    The significant part is defined relative to the maximum absolute value
    in the array. Elements A[:i1] and A[i2:] are considered "small".

    Args:
      array_data: The input 1D array.
      threshold: The relative threshold to determine significance.
                 Elements are significant if abs(element) >= threshold * max(abs(array_data)).

    Returns:
      A tuple (i1, i2) representing the start (inclusive) and
      end (exclusive) indices of the significant part.

    """
    abs_array = np.abs(array_data)
    if abs_array.size == 0:
        return 0, 0

    idx_max = np.argmax(abs_array)
    array_max_val = abs_array[idx_max]

    if array_max_val == 0:  # All elements are zero
        return 0, len(array_data)

    significant_indices_before_max = np.where(
        abs_array[:idx_max] >= threshold * array_max_val
    )[0]
    i1 = (
        significant_indices_before_max[0]
        if significant_indices_before_max.size > 0
        else idx_max
    )

    significant_indices_from_max = np.where(
        abs_array[idx_max:] >= threshold * array_max_val
    )[0]
    i2 = (
        idx_max + significant_indices_from_max[-1] + 1
        if significant_indices_from_max.size > 0
        else idx_max + 1
    )
    return i1, i2


def get_significant_part_indices_v2(
    array_data: ArrayLike, threshold: float = 1e-8
) -> Tuple[int, int]:
    """Returns indices (i1, i2) based on parts of the array that are "small".

    The interpretation of (i1, i2) from the original code is:
    `i1` is the index of the last element *before* the peak region that is
    considered small (abs(element) < threshold * max_val).
    `i2` is the index *after* the first element *after* the peak region that
    is considered small.
    The docstring of the original function was "Return a tuple (i1,i2) such
    that none of the elements A[i1:i2] is small", which might be misleading
    as A[i1] and A[i2-1] could themselves be small by this definition.
    A slice like A[i1+1 : i2-1] or A[i1+1 : idx_first_small_after_peak]
    might better correspond to "all elements are not small".

    Args:
      array_data: The input 1D array.
      threshold: The relative threshold to determine smallness.
                 Elements are small if abs(element) < threshold * max(abs(array_data)).

    Returns:
      A tuple (i1, i2).

    """
    abs_array = np.abs(array_data)
    if abs_array.size == 0:
        return 0, 0

    idx_max = np.argmax(abs_array)
    array_max_val = abs_array[idx_max]

    if array_max_val == 0:  # All elements are zero
        return 0, len(array_data)

    small_indices_before_max = np.where(
        abs_array[:idx_max] < threshold * array_max_val
    )[0]
    i1 = small_indices_before_max[-1] if small_indices_before_max.size > 0 else 0

    small_indices_from_max = np.where(abs_array[idx_max:] < threshold * array_max_val)[
        0
    ]
    # small_indices_from_max are relative to idx_max
    i2 = (
        idx_max + small_indices_from_max[0] + 1
        if small_indices_from_max.size > 0
        else len(array_data)
    )
    return i1, i2


def Fourier_filter(
    data: ArrayLike,
    time_step: float,
    spectral_window: ArrayLike,
    periodic: bool = False,
) -> ArrayLike:
    """Applies a Fourier filter to time-series data.

    The function performs a Fourier transform, multiplies by a spectral
    window, and then performs an inverse Fourier transform.

    Args:
      data: A 1D array or a 2D array (num_time_points, num_series)
            where each column is a time series.
      time_step: The time step between data points.
      spectral_window: A 2D array (num_window_points, 2) where the first
                       column contains circular frequencies (ascending) and
                       the second column contains the window function W(omega).
      periodic: If True, data is assumed to be periodic. Otherwise, data is
                mirrored and padded to reduce edge effects.

    Returns:
      The filtered data, with the same shape as the input `data`.

    """
    if spectral_window.shape[0] == 0:
        return data.copy()  # Return a copy to match behavior when filtering occurs

    original_shape = data.shape
    if data.ndim == 1:
        # Reshape 1D array to 2D for consistent processing
        current_data = data.reshape(-1, 1)
    else:
        current_data = data.copy()  # Work on a copy

    num_time_points = current_data.shape[0]

    if not periodic:
        # Mirror and concatenate data for non-periodic signals
        # Effectively doubles the number of time points for FFT
        current_data = np.vstack((current_data, current_data[::-1, :]))
        num_time_points_fft = current_data.shape[0]
    else:
        num_time_points_fft = num_time_points

    # Fourier transform
    Fourier_transformed_data = np.fft.fftshift(np.fft.fft(current_data, axis=0), axes=0)

    # Create frequency grid for the spectral window
    delta_omega = 2 * np.pi / (num_time_points_fft * time_step)
    # np.arange needs to create num_time_points_fft points
    # fftshift moves 0 frequency to the center.
    # The indices for fftshift range from -N/2 to N/2-1 (approx)
    omega_grid_indices = np.arange(num_time_points_fft) - np.floor(
        num_time_points_fft / 2.0
    )
    omega_grid = delta_omega * omega_grid_indices

    # Interpolate spectral window onto the data's frequency grid
    # Ensure omega_grid is 1D for interpolation if it became 2D due to reshape
    window_values = np.interp(
        np.abs(omega_grid.ravel()),  # Use absolute frequencies
        spectral_window[:, 0],
        spectral_window[:, 1],
        left=1.0,  # Value for frequencies below spectral_window range
        right=0.0,  # Value for frequencies above spectral_window range
    )

    # Apply spectral filter
    # window_values needs to be (num_time_points_fft, 1) to broadcast
    Fourier_transformed_data *= window_values.reshape(-1, 1)

    # Inverse Fourier transform
    filtered_data_full = np.fft.ifft(
        np.fft.ifftshift(Fourier_transformed_data, axes=0), axis=0
    )

    if not periodic:
        # Truncate to original length if data was mirrored
        filtered_data_final = filtered_data_full[:num_time_points, :]
    else:
        filtered_data_final = filtered_data_full

    # Ensure output is real if input was real
    if np.all(np.isreal(data)):
        filtered_data_final = filtered_data_final.real

    return filtered_data_final.reshape(original_shape)


def polyfit_with_weights(
    x_coords: ArrayLike, y_values: ArrayLike, weights: ArrayLike, degree: int
) -> ArrayLike:
    """Performs a weighted least-squares polynomial fit.

    Args:
      x_coords: 1D array of sample point x-coordinates.
      y_values: 1D array of sample point y-values to fit.
      weights: 1D array of weights for each sample point.
      degree: Degree of the fitting polynomial.

    Returns:
      A 1D array of polynomial coefficients [p_degree, ..., p_1, p_0].

    """
    num_coeffs = degree + 1
    matrix_a = np.empty((num_coeffs, num_coeffs), dtype=np.float64)
    weights_squared = weights * weights

    # Construct the Vandermonde-like matrix A for the normal equations
    # A[i,j] = sum(w_k^2 * x_k^(i+j))
    for i in range(num_coeffs):
        for j in range(i, num_coeffs):
            matrix_a[i, j] = np.sum(weights_squared * (x_coords ** (i + j)))
            if i != j:
                matrix_a[j, i] = matrix_a[i, j]  # Symmetric matrix

    # Construct the vector b for the normal equations
    # b[i] = sum(w_k^2 * y_k * x_k^i)
    vector_b = np.empty(num_coeffs, dtype=np.float64)
    for i in range(num_coeffs):
        vector_b[i] = np.sum(weights_squared * y_values * (x_coords**i))

    # Solve the linear system A * p = b for coefficients p
    solution_coeffs = scipy.linalg.solve(matrix_a, vector_b)
    return solution_coeffs[::-1]  # Return in conventional order (highest power first)


def Fourier_transform(
    time_points: np.ndarray,
    y_data: np.ndarray,
    target_frequencies: Optional[np.ndarray] = None,
    is_periodic: bool = False,
    pulse_center_times: Optional[Union[float, np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Apply the Fast Fourier Transform (FFT) in an easy-to-use way.

    This function computes the Fourier transform of time-dependent data Y(t),
    defined as:
        $F[Y](\omega) = \int_{-\infty}^\infty dt Y(t) \exp(i \omega t)$

    Args:
        time_points: A 1D NumPy array of shape (N_t,) representing the time
            discretization of Y(t). Must be sorted in ascending order. N_t must
            be greater than 1.
        y_data: A NumPy array of shape (N_t, N_data) or (N_t,) containing the
            time-dependent data to Fourier transform. If 1D, it's treated as
            a single data series.
        target_frequencies: An optional 1D NumPy array of shape (N_omega,)
            specifying the circular frequencies at which to compute the
            transform. Must be sorted in ascending order. If None, the transform
            is returned on an optimal internal frequency grid.
        is_periodic: A boolean indicating if Y(t) is assumed to be periodic.
            If True, the time step is expected to be constant, and
            Y(t_end + dt) = Y(t_start). If False, precautions are taken to
            minimize artifacts from the DFT's implicit periodicity assumption.
        pulse_center_times: An optional scalar or 1D NumPy array of length
            N_data. If the input data represents a pulse not centered in its
            time window, specifying the center of each pulse (t0) helps avoid
            interpolation artifacts. If None, the center of the time window is
            used.

    Returns:
        If `target_frequencies` is None:
            A tuple `(transformed_y, fft_omega_grid)`, where:
            - `transformed_y` (np.ndarray): The Fourier-transformed data,
              shape (N_fft, N_data) or (N_fft,) if input `y_data` was 1D.
            - `fft_omega_grid` (np.ndarray): The array of circular frequencies
              (shape (N_fft,)) corresponding to `transformed_y`.
        If `target_frequencies` is not None:
            An np.ndarray of shape (N_omega, N_data) or (N_omega,) containing
            the Fourier transform of `y_data` evaluated at the specified
            `target_frequencies`.

    Raises:
        ValueError: If `time_points` has fewer than 2 elements.

    """
    _MAX_FFT_POINTS = 2**20
    num_time_points = len(time_points)
    if num_time_points <= 1:
        raise ValueError("Input 'time_points' array must have more than one element.")

    original_y_ndim = y_data.ndim
    if original_y_ndim == 1:
        # Reshape 1D y_data to (N_t, 1) for consistent processing
        y_data_processed = y_data.reshape((num_time_points, 1))
    else:
        y_data_processed = y_data

    num_data_series = y_data_processed.shape[1]

    # Determine the minimum time step from the input time_points
    min_time_step = np.min(np.diff(time_points))
    # This will be the effective time step for FFT. It might be modified later
    # for non-periodic cases to meet Nyquist criteria for target_frequencies.
    effective_time_step = min_time_step

    period_duration: Optional[float] = None
    if is_periodic:
        # For periodic signals, the period is the total duration assuming
        # a constant step.
        period_duration = (
            time_points[-1] - time_points[0] + np.mean(np.diff(time_points))
        )

    # Handle the case where only one target frequency is given (no FFT needed)
    if target_frequencies is not None and len(target_frequencies) == 1:
        # Direct integration using trapezoidal rule
        # integrand shape: (num_time_points, num_data_series)
        integrand = y_data_processed * np.exp(
            1j * target_frequencies * time_points.reshape((num_time_points, 1))
        )
        # result shape: (num_data_series,)
        output_spectrum = np.trapezoid(integrand, time_points, axis=0)
        if is_periodic:
            # Correction for periodic functions with trapezoidal rule
            output_spectrum += (
                0.5 * min_time_step * (integrand[0, :] + integrand[-1, :])
            )
        # Reshape to (1, num_data_series) to match expected output shape
        output_spectrum = output_spectrum.reshape(1, num_data_series)
        if original_y_ndim == 1:
            return output_spectrum.reshape(len(target_frequencies))  # or .flatten()
        return output_spectrum

    # Determine the target frequency resolution for FFT grid calculation
    dw_target_for_n_fft_calc: float
    if target_frequencies is None:
        # If no target frequencies, FFT resolution is based on input time window
        dw_target_for_n_fft_calc = 2 * np.pi / (min_time_step * num_time_points)
    else:
        # If target frequencies are given, use their minimum spacing
        dw_target_for_n_fft_calc = np.min(np.diff(target_frequencies))

    # Determine the number of points for FFT (num_fft_points)
    num_fft_points: int
    if is_periodic:
        num_fft_points = num_time_points
        # If target frequencies are specified, may need to upsample N_fft
        # to achieve finer resolution than the default N_t points.
        # effective_time_step remains min_time_step for periodic case.
        if target_frequencies is not None:
            while (
                2 * np.pi / (effective_time_step * num_fft_points)
                > 1.1 * dw_target_for_n_fft_calc
                and num_fft_points < _MAX_FFT_POINTS
            ):
                num_fft_points *= 2
    else:  # Not periodic
        # Initial estimate for num_fft_points based on desired resolution
        num_fft_points = 2 ** int(
            round(
                np.log(2 * np.pi / (min_time_step * dw_target_for_n_fft_calc))
                / np.log(2.0)
            )
        )
        # Ensure FFT time window is large enough to cover the original data span
        while (num_fft_points - 1) * min_time_step < time_points[-1] - time_points[
            0
        ] and num_fft_points < _MAX_FFT_POINTS:
            num_fft_points *= 2

        # For non-periodic signals with specified target_frequencies,
        # adjust effective_time_step and num_fft_points to satisfy Nyquist
        # and resolution requirements for the target_frequencies.
        if target_frequencies is not None:
            # Use temporary variables for this iterative adjustment
            current_dt = min_time_step
            current_n_fft = num_fft_points
            while current_n_fft < _MAX_FFT_POINTS and (
                target_frequencies[-1]
                > np.pi / current_dt * (1.0 - 2.0 / current_n_fft)
                or -target_frequencies[0] > np.pi / current_dt
            ):
                current_dt /= 2.0
                current_n_fft *= 2
            effective_time_step = current_dt
            num_fft_points = current_n_fft

    # FFT time grid generation
    # This grid is centered around 0 after shifting by pulse_center_times
    fft_time_grid = effective_time_step * np.fft.ifftshift(
        np.arange(num_fft_points) - num_fft_points // 2
    )

    # Determine effective pulse center times for interpolation and phase shift
    # pulse_centers_for_interp: used to shift time_points before interpolation
    # pulse_centers_for_phase: used for final phase correction, shape (1, N_data)

    _pulse_centers_for_interp: Union[float, np.ndarray]
    _pulse_centers_for_phase: np.ndarray

    if pulse_center_times is None:
        # If no t0 provided, use the center of the time window
        time_window_center = 0.5 * (time_points[0] + time_points[-1])
        # Find the closest point in time_points to this center
        idx_center = np.argmin(np.abs(time_points - time_window_center))
        # This becomes the t0 for all data series
        calculated_t0 = time_points[idx_center]
        _pulse_centers_for_interp = calculated_t0  # Scalar
        _pulse_centers_for_phase = calculated_t0 * np.ones((1, num_data_series))
    else:
        _pulse_centers_for_interp = pulse_center_times  # Can be scalar or array
        if np.isscalar(pulse_center_times):
            _pulse_centers_for_phase = pulse_center_times * np.ones(
                (1, num_data_series)
            )
        else:
            # Ensure it's a 1D array before reshaping
            _pulse_centers_for_phase = np.asarray(pulse_center_times).reshape(
                (1, num_data_series)
            )

    # Interpolate y_data onto the FFT time grid
    # y_interpolated_on_fft_grid shape: (num_fft_points, num_data_series)
    y_interpolated_on_fft_grid = np.zeros(
        (num_fft_points, num_data_series), dtype=y_data_processed.dtype
    )

    for j_col in range(num_data_series):
        current_t0_interp: float
        if np.isscalar(_pulse_centers_for_interp):
            current_t0_interp = float(_pulse_centers_for_interp)
        else:
            current_t0_interp = float(np.asarray(_pulse_centers_for_interp)[j_col])

        # Shift original time_points relative to the current pulse center
        shifted_time_points = time_points - current_t0_interp

        if is_periodic:
            # For periodic data, use period in interpolation
            y_interpolated_on_fft_grid[:, j_col] = np.interp(
                fft_time_grid,
                shifted_time_points,
                y_data_processed[:, j_col],
                period=period_duration,
            )
        else:
            # For non-periodic, pad with zeros outside original time range
            y_interpolated_on_fft_grid[:, j_col] = np.interp(
                fft_time_grid,
                shifted_time_points,
                y_data_processed[:, j_col],
                left=0.0,
                right=0.0,
            )

    # Perform FFT
    # The result of ifft is scaled by (1/N). We multiply by (N*dt) to approximate
    # the integral definition F(omega) = integral Y(t)exp(i*omega*t) dt.
    # So, overall scaling is dt.
    y_fft = np.fft.fftshift(np.fft.ifft(y_interpolated_on_fft_grid, axis=0), axes=0) * (
        num_fft_points * effective_time_step
    )

    # FFT omega grid
    fft_omega_grid_spacing = 2 * np.pi / (effective_time_step * num_fft_points)
    fft_omega_grid = fft_omega_grid_spacing * (
        np.arange(num_fft_points) - num_fft_points // 2
    )

    # Apply phase correction due to pulse_center_times (t0)
    # This accounts for the shift Y(t) -> Y(t-t0) in time domain,
    # which corresponds to F(omega) -> F(omega) * exp(i*omega*t0)
    # if the FFT was performed on data effectively centered at t'=0.
    # The interpolation shifted data by -t0, so Z was Y(t').
    # The FFT of Y(t') is F[Y(t')] = integral Y(t')exp(iwt')dt'.
    # We want F[Y(t)] = integral Y(t)exp(iwt)dt.
    # F[Y(t)] = exp(iw t0_effective) * F[Y(t')].
    phase_correction = np.exp(
        1j * _pulse_centers_for_phase * fft_omega_grid.reshape((num_fft_points, 1))
    )
    y_fft_corrected = y_fft * phase_correction

    if target_frequencies is None:
        # Return FFT result on its own grid
        if original_y_ndim == 1:
            return y_fft_corrected.flatten(), fft_omega_grid
        return y_fft_corrected, fft_omega_grid
    # Interpolate FFT result onto the target_frequencies grid
    output_spectrum = np.zeros(
        (len(target_frequencies), num_data_series), dtype=np.complex128
    )
    for j_col in range(num_data_series):
        # Note: y_fft_corrected already includes the phase shift based on _pulse_centers_for_phase
        # and fft_omega_grid. When interpolating to target_frequencies, this phase is implicitly
        # interpolated as well.
        output_spectrum[:, j_col] = np.interp(
            target_frequencies,
            fft_omega_grid,
            y_fft_corrected[:, j_col],  # Use the phase-corrected FFT result
            left=0.0,
            right=0.0,
        )

    # The phase correction was already applied to y_fft before interpolation.
    # If we were to apply it *after* interpolation, it would be:
    # phase_correction_on_target_freq = np.exp(
    #    1j * _pulse_centers_for_phase * target_frequencies.reshape((len(target_frequencies), 1))
    # )
    # output_spectrum = interpolated_unphased_result * phase_correction_on_target_freq
    # However, the original code applies the phase correction *before* this final interpolation step
    # if omega is None, and *after* if omega is not None.
    # Let's re-check original logic for omega not None:
    # Z = np.fft.fftshift(np.fft.ifft(Z, axis=0), axes=0) * (N_fft * dt) <-- y_fft (unphased by t0 yet for this path)
    # ...
    # result[:,j] = np.interp(omega, w_grid, Z[:,j], left=0.0, right=0.0) <-- interpolation of unphased
    # result = result * np.exp(1j * t0 * omega.reshape((len(omega), 1))) <-- phase correction
    # This means my current y_fft_corrected (which has phase) should NOT be used for interpolation here.
    # I should interpolate 'y_fft' (before t0 correction) and then apply t0 correction using target_frequencies.

    # Reverting to match original logic for target_frequencies path:
    # Interpolate the raw FFT result (before t0 correction)
    interpolated_raw_fft = np.zeros(
        (len(target_frequencies), num_data_series), dtype=np.complex128
    )
    for j_col in range(num_data_series):
        interpolated_raw_fft[:, j_col] = np.interp(
            target_frequencies,
            fft_omega_grid,
            y_fft[:, j_col],  # Use y_fft (before _pulse_centers_for_phase correction)
            left=0.0,
            right=0.0,
        )

    # Now apply phase correction using _pulse_centers_for_phase and target_frequencies
    phase_correction_final = np.exp(
        1j
        * _pulse_centers_for_phase
        * target_frequencies.reshape((len(target_frequencies), 1))
    )
    output_spectrum = interpolated_raw_fft * phase_correction_final

    if original_y_ndim == 1:
        return output_spectrum.flatten()
    return output_spectrum


def inverse_Fourier_transform(
    omega_points: ArrayLike,
    data_series: ArrayLike,
    time_points_target: Optional[ArrayLike] = None,
    is_periodic: bool = False,
    frequency_offset: Optional[Union[float, ArrayLike]] = None,
) -> Union[Tuple[ArrayLike, ArrayLike], ArrayLike]:
    r"""Apply inverse FFT to frequency-dependent data.

    Computes $ F^{-1}[Y](t) = 1 / (2 \pi) \int_{-\infty}^\infty d\omega Y(\omega) \exp(-i t \omega) $.

    Args:
      omega_points: 1D array of circular frequencies (N_omega), sorted.
      data_series: Frequency-dependent data (N_omega) or (N_omega, N_data_series).
      time_points_target: Optional 1D array of time points (N_t), sorted.
                          If None, times are determined by IFFT.
      is_periodic: If True, data_series is assumed periodic in frequency.
      frequency_offset: Scalar or array (N_data_series). Central frequency
                        offset(s) if data is not centered at omega=0.

    Returns:
      If time_points_target is None:
        A tuple (transformed_data, time_grid).
      If time_points_target is provided:
        Transformed data interpolated at the given time_points_target.

    """
    # IFFT(Y(w)) = 1/(2pi) FT(Y(w))_at_-t = 1/(2pi) conj(FT(conj(Y(w)))_at_t)
    # The provided Fourier_transform computes FT[Y(t)](omega) = integral Y(t) exp(iwt) dt
    # We want IFT[Y(w)](t) = 1/(2pi) integral Y(w) exp(-iwt) dw
    # Let w' = -w, dw' = -dw.
    # = -1/(2pi) integral Y(-w') exp(iw't) dw' (from -inf to +inf, so limits flip)
    # = 1/(2pi) integral Y(-w') exp(iw't) dw' (from -inf to +inf)
    # So, call Fourier_transform with omega -> -omega (reversed), Y(omega) -> Y(-omega) (reversed)
    # and then scale by 1/(2pi). The 't' in Fourier_transform becomes our 'omega_points',
    # and 'omega' in Fourier_transform becomes our '-time_points_target'.

    if time_points_target is None:
        # Transform Y(omega) as if it's a time signal Y(t=omega)
        # The 'omega' output of Fourier_transform will correspond to '-t'
        transformed_data, neg_time_grid = Fourier_transform(
            time_points=omega_points,
            y_data=data_series,
            target_frequencies=None,  # Let FT determine output grid
            is_periodic=is_periodic,
            pulse_center_times=frequency_offset,  # This is omega0, an offset in the input "time" (omega) domain
        )
        # Result is FT[Y](k), where k is frequency. Here k corresponds to -t.
        # So, FT[Y(omega)](-t). We need to flip t and scale.
        return transformed_data[::-1] / (2 * np.pi), -neg_time_grid[::-1]
    # Target 'omega' for Fourier_transform is -time_points_target
    neg_target_times = -time_points_target[::-1]  # Ensure it's sorted for FT

    result_at_neg_t = Fourier_transform(
        time_points=omega_points,
        y_data=data_series,
        target_frequencies=neg_target_times,
        is_periodic=is_periodic,
        pulse_center_times=frequency_offset,
    )
    # result_at_neg_t is FT[Y(omega)](-t_target_sorted)
    # We want values at t_target, so reverse the order back.
    return result_at_neg_t[::-1] / (2 * np.pi)


def find_zero_crossings(x_values: ArrayLike, y_values: ArrayLike) -> ArrayLike:
    """Find all x-values where linearly interpolated y(x) = 0.

    Args:
      x_values: 1D array of x-coordinates, sorted ascending, no duplicates.
      y_values: 1D array of y-coordinates, same shape as x_values.

    Returns:
      A 1D array of x-values where y(x) crosses zero. Empty if no crossings.

    """
    if x_values.size == 0 or y_values.size == 0:
        return np.array([])
    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have the same length.")

    # Product of y[i] and y[i+1]
    product_adjacent_y = y_values[:-1] * y_values[1:]
    crossings_x_coords: List[float] = []

    # Find indices where product is <= 0 (indicates a zero crossing or y[i]=0)
    for i in np.where(product_adjacent_y <= 0)[0]:
        # Instead of: if product_adjacent_y[i] == 0:
        #                 if y_values[i] == 0:
        # Use np.isclose for checking if y_values[i] or y_values[i+1] are zero
        y1_is_zero = np.isclose(y_values[i], 0.0)
        y2_is_zero = np.isclose(y_values[i + 1], 0.0)

        if y1_is_zero and y2_is_zero:  # segment is [0,0]
            crossings_x_coords.append(x_values[i])
            # To avoid double adding x_values[i+1] if it's processed as y1_is_zero in next iter
        elif y1_is_zero:
            crossings_x_coords.append(x_values[i])
        elif (
            y2_is_zero and product_adjacent_y[i] < 0
        ):  # Crosses and lands on zero at y2
            # The interpolation formula will give x_values[i+1]
            x1, x2_pt = x_values[i], x_values[i + 1]
            y1_pt, y2_pt = y_values[i], y_values[i + 1]  # y2_pt is close to 0
            crossings_x_coords.append((x1 * y2_pt - x2_pt * y1_pt) / (y2_pt - y1_pt))
        elif product_adjacent_y[i] < 0:  # Definite crossing, neither is zero
            x1, x2 = x_values[i], x_values[i + 1]
            y1_val, y2_val = y_values[i], y_values[i + 1]
            crossings_x_coords.append((x1 * y2_val - x2 * y1_val) / (y2_val - y1_val))

    # Handle case where the last point itself is a zero not caught by pair product
    # This also needs np.isclose
    if y_values.size > 0 and np.isclose(y_values[-1], 0.0):
        # Avoid adding if it's already part of a segment ending in zero
        # that was captured by product_adjacent_y[i]=0 logic (where y[i+1]=0)
        already_found = False
        if crossings_x_coords and np.isclose(crossings_x_coords[-1], x_values[-1]):
            already_found = True

        if not already_found:
            # If y_values[-1] is zero, and y_values[-2]*y_values[-1] was not <=0 (e.g. y_values[-2] also zero)
            # or it was handled by interpolation which might be slightly off.
            # We want to ensure grid points that are zero are included.
            # A simpler way: collect all interpolated, then add all x where y is zero, then unique.
            pass  # The unique call later should handle it if x_values[-1] was added by main loop

    # A more robust approach for points exactly on the grid:
    # After interpolation, add all x_values where corresponding y_values are close to zero.
    if x_values.size > 0:  # Ensure x_values is not empty
        grid_zeros = x_values[np.isclose(y_values, 0.0)]
        crossings_x_coords.extend(list(grid_zeros))

    return np.unique(np.array(crossings_x_coords))


def find_extrema_positions(x_values: ArrayLike, y_values: ArrayLike) -> ArrayLike:
    """Find x-positions of local extrema in y(x).

    Extrema are found where the derivative y'(x) (approximated by finite
    differences) crosses zero.

    Args:
      x_values: 1D array of x-coordinates, sorted ascending, no duplicates.
      y_values: 1D array of y-coordinates, same shape as x_values.

    Returns:
      A 1D array of x-values where y(x) has local extrema. Empty if none.

    """
    if (
        len(x_values) < 2 or len(y_values) < 2
    ):  # Need at least two points for a derivative
        return np.array([])
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    # Approximate derivative y'(x)
    delta_y = y_values[1:] - y_values[:-1]
    delta_x = x_values[1:] - x_values[:-1]
    # Avoid division by zero if x_values have duplicates (though pre-condition says no duplicates)
    # However, if delta_x is extremely small, derivative can be huge.
    # For robustness, filter out zero delta_x if they somehow occur.
    valid_dx = delta_x != 0
    if not np.all(valid_dx):  # Should not happen given preconditions
        delta_y = delta_y[valid_dx]
        delta_x = delta_x[valid_dx]
        mid_points_x_for_derivative = (x_values[1:] + x_values[:-1])[valid_dx] / 2.0
    else:
        mid_points_x_for_derivative = (x_values[1:] + x_values[:-1]) / 2.0

    if delta_x.size == 0:  # Not enough points after filtering
        return np.array([])

    derivative_y = delta_y / delta_x

    # Find where the derivative crosses zero
    return find_zero_crossings(mid_points_x_for_derivative, derivative_y)


def minimize_imaginary_parts(complex_array: ArrayLike) -> ArrayLike:
    """Rotates a complex array by a phase to make it as close as possible to being real-valued.

    Multiplies `complex_array` by `exp(1j*phi)` where `phi` is chosen to
    minimize `sum(imag(exp(1j*phi) * complex_array)**2)`.

    Args:
      complex_array: A NumPy array of complex numbers.

    Returns:
      The phase-rotated complex NumPy array.

    """
    if complex_array.size == 0:
        return complex_array.copy()

    # Z = X + iY. We want to minimize sum( (X sin(phi) + Y cos(phi))^2 )
    # d/dphi (sum(...)) = 0 leads to tan(2*phi) = 2*sum(XY) / sum(Y^2 - X^2)
    real_part = complex_array.real
    imag_part = complex_array.imag

    numerator = 2 * np.sum(real_part * imag_part)
    denominator = np.sum(imag_part**2 - real_part**2)

    # arctan2 handles signs and denominator being zero correctly
    phi = 0.5 * np.arctan2(numerator, denominator)

    # The arctan2 gives phi in (-pi, pi], so 0.5*phi is in (-pi/2, pi/2].
    # This finds one extremum. The other is phi + pi/2. We need the minimum.
    rotated_z1 = complex_array * np.exp(1j * phi)
    imag_energy1 = np.sum(rotated_z1.imag**2)

    rotated_z2 = complex_array * np.exp(1j * (phi + 0.5 * np.pi))
    imag_energy2 = np.sum(rotated_z2.imag**2)

    if imag_energy2 < imag_energy1:
        phi += 0.5 * np.pi

    # Normalize phi to be in (-pi/2, pi/2] or a similar principal range if desired,
    # though for exp(1j*phi) it doesn't strictly matter beyond 2pi periodicity.
    # The original code maps phi to (-pi/2, pi/2] effectively.
    phi -= np.pi * np.round(phi / np.pi)  # This maps to (-pi/2, pi/2]
    # Let's test the original normalization:
    # If phi = 0.6*pi, round(0.6) = 1. phi = 0.6pi - pi = -0.4pi. Correct.
    # If phi = 0.4*pi, round(0.4) = 0. phi = 0.4pi. Correct.
    # If phi = -0.6*pi, round(-0.6) = -1. phi = -0.6pi + pi = 0.4pi. Correct.
    # This normalization is fine.

    return complex_array * np.exp(1j * phi)


def integrate_oscillating_function(
    x_values: ArrayLike,
    func_values: ArrayLike,
    phase_values: ArrayLike,
    phase_step_threshold: float = 1e-3,
) -> ArrayLike:
    r"""Integrates f(x) * exp(i * phi(x)) for quickly oscillating functions.

    Uses an algorithm suitable for integrating f(x) * exp(i * phi(x)) dx
    over small intervals, particularly when phi(x) changes rapidly.

    Args:
      x_values: 1D array of sorted x-coordinates.
      func_values: Array of function values f(x). Can be 1D (N_x) or 2D
                   (N_x, N_series).
      phase_values: Array of real-valued phase phi(x). Same shape as func_values.
      phase_step_threshold: Small positive number. Prevents division by
                            small d_phi in the integration formula.

    Returns:
      A scalar or 1D array (N_series) of integral results.

    """
    # Input validation
    if not (x_values.shape[0] == func_values.shape[0] == phase_values.shape[0]):
        raise ValueError(
            "x_values, func_values, and phase_values must have "
            "the same length along the integration axis (axis 0)."
        )
    if not np.allclose(np.imag(phase_values), 0):  # Ensure phase is real
        raise ValueError("phase_values must be real-valued.")
    if func_values.ndim > 1 and func_values.shape != phase_values.shape:
        raise ValueError(
            "If func_values is 2D, phase_values must have the exact same shape."
        )

    delta_x = x_values[1:] - x_values[:-1]

    # Prepare f and phi for interval calculations
    f1 = func_values[:-1, ...]  # f(x_i)
    f2 = func_values[1:, ...]  # f(x_{i+1})
    delta_f = f2 - f1

    phi1 = phase_values[:-1, ...]  # phi(x_i)
    phi2 = phase_values[1:, ...]  # phi(x_{i+1})
    delta_phi = phi2 - phi1

    # Reshape delta_x to broadcast with f1, f2, etc.
    # If func_values is (N_x, N_series), delta_x needs to be (N_x-1, 1)
    reshape_dims = (-1,) + (1,) * (func_values.ndim - 1)
    delta_x_reshaped = delta_x.reshape(reshape_dims)

    # Common factor for the integral segments
    common_factor_z = delta_x_reshaped * np.exp(0.5j * (phi1 + phi2))

    integral_segments = np.zeros_like(common_factor_z, dtype=complex)

    # Mask for small phase changes (use simpler approximation)
    is_small_delta_phi = np.abs(delta_phi) < phase_step_threshold

    # Case 1: Small delta_phi (dphi is small)
    if np.any(is_small_delta_phi):
        # Approximation: integral \approx dx * exp(i*phi_avg) * (f_avg + i/8 * dphi * df)
        # This seems to be a higher-order trapezoidal rule for oscillating functions.
        # Original: Z[s] = Z[s] * (0.5 * (f1[s] + f2[s]) + 0.125j * dphi[s] * df[s])
        # where Z[s] was common_factor_z[is_small_delta_phi]
        term_small_dphi = (
            0.5 * (f1[is_small_delta_phi] + f2[is_small_delta_phi])
            + 0.125j * delta_phi[is_small_delta_phi] * delta_f[is_small_delta_phi]
        )
        integral_segments[is_small_delta_phi] = (
            common_factor_z[is_small_delta_phi] * term_small_dphi
        )

    # Case 2: Large delta_phi (use formula for oscillating part)
    is_large_delta_phi = ~is_small_delta_phi
    if np.any(is_large_delta_phi):
        # This is likely an approximation based on integration by parts or steepest descent.
        # Original: Z[s] = Z[s] / dphi[s]**2 * (exp_term * (df[s] - 1j*f2[s]*dphi[s]) -
        #                                     (df[s] - 1j*f1[s]*dphi[s]) / exp_term)
        # where Z[s] was common_factor_z[is_large_delta_phi] and exp_term = exp(0.5j * dphi[s])

        dphi_large = delta_phi[is_large_delta_phi]
        exp_half_j_dphi = np.exp(0.5j * dphi_large)

        term1 = exp_half_j_dphi * (
            delta_f[is_large_delta_phi] - 1j * f2[is_large_delta_phi] * dphi_large
        )
        term2 = (
            delta_f[is_large_delta_phi] - 1j * f1[is_large_delta_phi] * dphi_large
        ) / exp_half_j_dphi

        integral_segments[is_large_delta_phi] = (
            common_factor_z[is_large_delta_phi] / (dphi_large**2) * (term1 - term2)
        )

    return np.sum(integral_segments, axis=0)


def calculate_permittivity_from_delta_polarization(
    time_step: float,
    polarization_delta_response: ArrayLike,  # P_delta
    omega_array: ArrayLike,
    momentum_relaxation_rate: float = 0.0,
    dephasing_time: Optional[float] = None,
    disregard_drift_current: bool = False,
    allow_for_linear_displacement: bool = True,
) -> ArrayLike:
    r"""Evaluate permittivity from polarization induced by E(t) = delta(t).

    Handles drift currents and coherent oscillations in the polarization response.
    The relationship is $\epsilon(\omega) = 1 + 4 \pi \chi(\omega)$, where
    $P(\omega) = \chi(\omega) E(\omega)$, and for $E(t)=\delta(t)$, $E(\omega)=1$.
    So $\chi(\omega) = P_{\delta}(\omega)$.

    Args:
      time_step: Time step (atomic units) of the polarization grid.
      polarization_delta_response: 1D array of polarization response P(t)
                                   (atomic units) induced by E(t)=delta(t).
                                   P_delta[0] corresponds to t=time_step.
      omega_array: 1D array of circular frequencies (a.u.) for permittivity calculation.
                   All frequencies must be non-zero.
      momentum_relaxation_rate: If non-zero, models Drude-like momentum relaxation
                                (gamma in 1/(omega*(omega+i*gamma))).
      dephasing_time: If not None/zero, an exponential decay (rate 1/dephasing_time)
                      is applied to coherent dipole oscillations in P_delta.
      disregard_drift_current: If True, the J_drift component of polarization
                               has no effect on the result.
      allow_for_linear_displacement: If True, fits P(t) ~ J_drift*t + P_offset.
                                     If False, fits P(t) ~ J_drift*t (P_offset=0).

    Returns:
      A complex array (same shape as omega_array) of permittivity values.

    """
    if not np.all(omega_array != 0):
        raise ValueError("All elements in omega_array must be non-zero.")

    # Construct time grid and full polarization array P(t), P(0)=0
    num_p_delta_points = polarization_delta_response.size
    # P_delta starts at t=dt, so P has N_t = num_p_delta_points + 1 points
    num_time_points = num_p_delta_points + 1
    time_grid = time_step * np.arange(num_time_points)
    time_max = time_grid[-1]

    polarization_full = np.zeros(num_time_points)
    polarization_full[1:] = polarization_delta_response  # P(0)=0

    # Fit and subtract linear trend (drift current and offset)
    # Fit is done on the latter half of the data
    fit_start_index = num_time_points // 2
    if (
        fit_start_index < 2 and num_time_points >= 2
    ):  # Need at least 2 points for polyfit(deg=1)
        fit_start_index = 0  # Use all data if too short for half
    elif num_time_points < 2:
        # Handle very short P_delta (e.g. 0 or 1 point)
        # If P_delta has 0 points, N_t=1, P_full=[0]. J_drift=0, P_offset=0.
        # If P_delta has 1 point, N_t=2, P_full=[0, P_d[0]].
        # polyfit needs at least deg+1 points.
        if num_time_points < 2:  # Cannot do polyfit
            J_drift = 0.0
            P_offset = 0.0
        elif allow_for_linear_displacement:  # N_t >= 2
            poly_coeffs = np.polyfit(
                time_grid[fit_start_index:], polarization_full[fit_start_index:], 1
            )
            J_drift = poly_coeffs[0]
            P_offset = poly_coeffs[1]
        else:  # N_t >= 2, P_offset = 0
            # P(t) = J_drift * t => J_drift = sum(P*t) / sum(t^2)
            # Ensure denominator is not zero if time_grid[fit_start_index:] is all zeros
            # (e.g., if fit_start_index is past end, or time_grid is [0,0,...])
            t_fit = time_grid[fit_start_index:]
            sum_t_squared = np.sum(t_fit**2)
            if sum_t_squared == 0:
                J_drift = 0.0
            else:
                J_drift = (
                    np.sum(polarization_full[fit_start_index:] * t_fit) / sum_t_squared
                )
            P_offset = 0.0
    else:  # Standard case N_t >= 2 and fit_start_index allows polyfit
        if allow_for_linear_displacement:
            poly_coeffs = np.polyfit(
                time_grid[fit_start_index:], polarization_full[fit_start_index:], 1
            )
            J_drift = poly_coeffs[0]
            P_offset = poly_coeffs[1]
        else:
            t_fit = time_grid[fit_start_index:]
            sum_t_squared = np.sum(t_fit**2)
            if sum_t_squared == 0:
                J_drift = 0.0
            else:
                J_drift = (
                    np.sum(polarization_full[fit_start_index:] * t_fit) / sum_t_squared
                )
            P_offset = 0.0

    # Subtract the J_drift * t part from polarization_full. P_offset remains for now.
    polarization_oscillating = polarization_full - J_drift * time_grid

    # Apply dephasing/windowing to the oscillating part (P - J_drift*t)
    # P_offset is part of the "DC" or very slow component, window it too.
    # The original code did: P_offset + window * (P - P_offset)
    # where P was (P_original - J_drift*t).
    # So, effectively: P_offset + window * (P_original - J_drift*t - P_offset)

    if dephasing_time is None or dephasing_time == 0:
        # Soft window to zero if no explicit dephasing
        time_window = soft_window(time_grid, 0.5 * time_max, time_max)
    else:
        time_window = np.exp(-time_grid / dephasing_time) * soft_window(
            time_grid, 0.5 * time_max, time_max
        )

    # Windowed polarization: P_offset is the value it decays from/to at t=0,
    # and the oscillating part (P_orig - J_drift*t - P_offset) is damped.
    processed_polarization = P_offset + time_window * (
        polarization_oscillating - P_offset
    )

    permittivity_results = np.zeros_like(omega_array, dtype=complex)

    for i, omega_val in enumerate(omega_array):
        # chi(omega) = FT[P_processed(t)](omega)
        # P_processed = P_offset_non_windowed + window * (P_osc - P_offset_non_windowed)
        # FT[P_processed] = FT[P_offset] + FT[window * (P_osc - P_offset)]
        # The original code integrated `processed_polarization` which is
        # P_offset + window * (P_original - J_drift*t - P_offset)

        chi_omega = integrate_oscillating_function(
            time_grid, processed_polarization, omega_val * time_grid
        )

        # Add analytical FT of the P_offset tail (if P_offset was not windowed to zero)
        # The `processed_polarization` already includes P_offset, partly windowed.
        # The original code adds: P_offset * 1j * np.exp(1j * omega * t_max) / omega
        # This looks like the FT of P_offset * Heaviside(t) if it extended from 0 to t_max
        # and then was abruptly cut, or FT of P_offset for t>t_max if window brought it to P_offset at t_max.
        # If processed_polarization(t_max) -> P_offset (due to window(t_max)=1),
        # and we assume P(t) = P_offset for t > t_max, its FT is P_offset * exp(i*omega*t_max) * (pi*delta(omega) + 1/(i*omega))
        # This term is tricky. The original `integrate_oscillating_function` handles up to t_max.
        # If the window makes processed_polarization(t_max) close to P_offset,
        # and we assume P(t) = P_offset for t > t_max, the integral from t_max to inf is
        # P_offset * integral_{t_max to inf} exp(i*omega*t) dt
        # = P_offset * [exp(i*omega*t) / (i*omega)]_{t_max to inf}
        # For convergence, Im(omega) > 0 or add damping. Assuming real omega, this diverges.
        # The term P_offset * 1j * np.exp(1j * omega * t_max) / omega
        # is -P_offset * exp(1j*omega*t_max) / (1j*omega). This is the upper limit of the integral.
        # It implies the lower limit (at infinity) is taken as zero.
        # This is the FT of P_offset * step_function(t_max - t) if integrated from -inf to t_max.
        # Or FT of P_offset for t > t_max, i.e. integral from t_max to infinity of P_offset*exp(iwt)
        # = P_offset * [exp(iwt)/(iw)]_{t_max to inf}. For this to be -P_offset*exp(iw*t_max)/(iw),
        # the exp(iw*inf) term must vanish, e.g. by small positive Im(w).
        # Let's assume this term correctly accounts for the P_offset tail beyond t_max.
        if not np.isclose(P_offset, 0.0):  # Only add if P_offset is significant
            chi_omega += P_offset * 1j * np.exp(1j * omega_val * time_max) / omega_val

        # Add contribution from drift current J_drift / (omega * (omega + i*gamma_relax))
        if not disregard_drift_current and not np.isclose(J_drift, 0.0):
            denominator_drift = omega_val * (omega_val + 1j * momentum_relaxation_rate)
            if not np.isclose(
                denominator_drift, 0.0
            ):  # Avoid division by zero if omega=0 (already checked) or omega=-i*gamma
                chi_omega -= J_drift / denominator_drift
            # If denominator is zero (e.g. omega_val=0 or omega_val = -1j*gamma), this term is singular.
            # omega_val != 0 is asserted. If omega_val = -1j*gamma, it's a resonant condition.

        permittivity_results[i] = 1.0 + 4 * np.pi * chi_omega

    return permittivity_results
