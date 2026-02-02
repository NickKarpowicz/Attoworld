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
    phi -= np.pi * np.round(phi / np.pi)  # This maps to (-pi/2, pi/2]

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

def coherence_time(Y: ArrayLike, dt: float = 1.0) -> float:
    r"""
    Given a signal Y(t) on an equidistant grid, the function estimates the coherence time.

    The function implements Mandel's coherence time. It calculates the autocorrelation
    function of the given signal, then it calculates the complex envelope of the
    autocorrelation function, and integrates the square of the normalized envelope from
    t=0 to the maximal time. The coherence time is twice the value of the integral.

    Args:
      Y: 1D array that represents the time-dependent signal on a regular grid.
      dt: Time step of the grid.

    Returns:
      the coherence time as a floating-point number.
    """
    ACF = scipy.signal.correlate(Y, Y, mode="full", method="direct") # autocorrelation function
    N = len(Y)
    # X = dt * np.arange(-(N - 1), N) # DEBUGGING
    ACF_envelope = np.abs(scipy.signal.envelope(ACF, residual=None))
    ACF_envelope /= np.max(ACF_envelope)
    i1 = N - 1 # np.flatnonzero(X >= 0)[0]
    # print("X[i1] =", X[i1]) # DEBUGGING
    return 2.0 * scipy.integrate.simpson(ACF_envelope[i1:]**2, dx=dt)
