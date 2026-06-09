use pyo3::prelude::*;
pub mod frog;
pub mod stencil;

/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
pub mod attoworld_rs {
    #[pymodule_export]
    pub use crate::frog::FrogType;
    use crate::frog::reconstruct_frog;
    use crate::stencil::{
        derivative, derivative_periodic, find_first_intercept, find_last_intercept,
        find_maximum_location, fornberg_stencil, interpolate_sorted_1d_slice, sort_paired_xy,
    };
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, ToPyArray};
    use pyo3::prelude::*;
    use rustfft::num_complex::Complex64;

    /// Find the location and value of the maximum of a smooth, uniformly sampled signal, interpolating to find the sub-pixel location
    ///
    /// Args:
    ///     y (np.ndarray): The signal whose maximum should be located
    ///     neighbors (int): the number of neighboring points to consider in the optimization (default 3)
    ///
    /// Returns:
    ///     (float, float): location, interpolated maximum
    #[pyfunction]
    #[pyo3(name = "find_maximum_location")]
    #[pyo3(signature = (y, neighbors = 3, /))]
    fn find_maximum_location_wrapper(
        y: PyReadonlyArrayDyn<f64>,
        neighbors: i64,
    ) -> PyResult<(f64, f64)> {
        match find_maximum_location(y.as_slice()?, neighbors) {
            Ok(result) => Ok(result),
            Err(()) => Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No maximum value possible; does the array contain a NaN value?",
            )),
        }
    }

    /// Find the first intercept with a value
    /// Args:
    ///     y (np.ndarray): the distribution data
    ///     intercept_value (float): The value at which to take the intercept
    ///     neighbors (int): The number of neighboring points in each direction to use when constructing interpolants. Higher values are more accurate, but only for smooth data.
    /// Returns:
    ///     float: "index" of the intercept, a float with non-integer value, indicating where between the pixels the intercept is
    #[pyfunction]
    #[pyo3(name = "find_first_intercept")]
    fn find_first_intercept_wrapper(
        y: PyReadonlyArrayDyn<f64>,
        intercept_value: f64,
        neighbors: usize,
    ) -> PyResult<f64> {
        Ok(find_first_intercept(
            y.as_slice()?,
            intercept_value,
            neighbors,
        ))
    }

    /// Find the last intercept with a value
    /// Args:
    ///     y (np.ndarray): the distribution data
    ///     intercept_value (float): The value at which to take the intercept
    ///     neighbors (int): The number of neighboring points in each direction to use when constructing interpolants. Higher values are more accurate, but only for smooth data.
    /// Returns:
    ///     float: "index" of the intercept, a float with non-integer value, indicating where between the pixels the intercept is
    #[pyfunction]
    #[pyo3(name = "find_last_intercept")]
    fn find_last_intercept_wrapper(
        y: PyReadonlyArrayDyn<f64>,
        intercept_value: f64,
        neighbors: usize,
    ) -> PyResult<f64> {
        Ok(find_last_intercept(
            y.as_slice()?,
            intercept_value,
            neighbors,
        ))
    }

    /// Find the full-width-at-half-maximum value of a continuously-spaced distribution.
    ///
    /// Args:
    ///     y (np.ndarray): the distribution data
    ///     dx (float): the x step size of the data
    ///     intercept_value (float): The value at which to take the intercepts (i.e. only full-width-at-HALF-max for 0.5)
    ///     neighbors (int): The number of neighboring points in each direction to use when constructing interpolants. Higher values are more accurate, but only for smooth data.
    /// Returns:
    ///     float: The full width at intercept_value maximum
    #[pyfunction]
    #[pyo3(name = "fwhm")]
    #[pyo3(signature = (y, dx = 1.0, intercept_value = 0.5, neighbors = 2))]
    fn fwhm(
        y: PyReadonlyArrayDyn<f64>,
        dx: f64,
        intercept_value: f64,
        neighbors: usize,
    ) -> PyResult<f64> {
        let (_, max_value) = match find_maximum_location(y.as_slice()?, neighbors as i64) {
            Ok(value) => value,
            Err(_) => {
                return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No maximum value possible; does the array contain a NaN value?",
                ));
            }
        };
        let first_intercept =
            find_first_intercept(y.as_slice()?, max_value * intercept_value, neighbors);
        let last_intercept =
            find_last_intercept(y.as_slice()?, max_value * intercept_value, neighbors);
        if first_intercept > last_intercept {
            println!(
                "Warning: internal calculation give a negative width, data may be too coarse to be reliable."
            );
        }
        Ok((dx * (last_intercept - first_intercept)).abs())
    }

    /// Generate a finite difference stencil using the algorithm described by B. Fornberg
    /// in Mathematics of Computation 51, 699-706 (1988).
    ///
    /// Args:
    ///     order (int): the order of the derivative
    ///     positions (np.ndarray): the positions at which the functions will be evaluated in the stencil. Must be larger than 2 elements in size.
    ///     position_out (float): the position at which using the stencil will evaluate the derivative, default 0.0.
    /// Returns:
    ///     np.ndarray: the finite difference stencil with weights corresponding to the positions in the positions input array
    ///
    /// Examples:
    ///
    ///     >>> stencil = fornberg_stencil(1, np.array([-1.0, 0.0, 1.0]))
    ///     >>> print(stencil)
    ///     [-0.5  0.   0.5]
    #[pyfunction]
    #[pyo3(name = "fornberg_stencil")]
    #[pyo3(signature = (order, positions, position_out = 0.0, /))]
    fn fornberg_stencil_wrapper<'py>(
        py: Python<'py>,
        order: usize,
        positions: PyReadonlyArrayDyn<'py, f64>,
        position_out: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(fornberg_stencil(order, positions.as_slice()?, position_out).into_pyarray(py))
    }

    /// Interpolate sorted data, given a list of intersection locations
    ///
    /// Args:
    ///     x_out (np.ndarray): array of output x values, the array onto which y_in will be interpolated
    ///     x_in (np.ndarray): array of input x values
    ///     y_in (np.ndarray): array of input y values
    ///     inputs_are_sorted (bool): true is x_in values are in ascending order (default). Set to false for unsorted data.
    ///     neighbors (int): number of nearest neighbors to include in the interpolation
    ///     extrapolate (bool): unless set to true, values outside of the range of x_in will be zero
    ///     derivative_order(int): order of derivative to take. 0 (default) is plain interpolation, 1 takes first derivative, and so on.
    ///
    /// Returns:
    ///     np.ndarray: the interpolated y_out
    #[pyfunction]
    #[pyo3(signature = (x_out, x_in, y_in,/, inputs_are_sorted=true, neighbors=2, extrapolate=false, derivative_order=0))]
    fn interpolate<'py>(
        py: Python<'py>,
        x_out: PyReadonlyArrayDyn<'py, f64>,
        x_in: PyReadonlyArrayDyn<'py, f64>,
        y_in: PyReadonlyArrayDyn<'py, f64>,
        inputs_are_sorted: bool,
        neighbors: i64,
        extrapolate: bool,
        derivative_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if inputs_are_sorted {
            Ok(interpolate_sorted_1d_slice(
                x_out.as_slice()?,
                x_in.as_slice()?,
                y_in.as_slice()?,
                neighbors,
                extrapolate,
                derivative_order,
            )
            .into_pyarray(py))
        } else {
            let (x_in_sorted, y_in_sorted) = sort_paired_xy(x_in.as_slice()?, y_in.as_slice()?);
            Ok(interpolate_sorted_1d_slice(
                x_out.as_slice()?,
                &x_in_sorted,
                &y_in_sorted,
                neighbors,
                extrapolate,
                derivative_order,
            )
            .into_pyarray(py))
        }
    }

    /// Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
    /// by using modified stencils that only use internal points.
    ///
    /// Args:
    ///     data (np.ndarray): the data whose derivative should be taken
    ///     order (int): the order of the derivative
    ///     neighbors (int): the number of nearest neighbors to consider in each direction.
    /// Returns:
    ///     np.ndarray: the derivative
    #[pyfunction]
    #[pyo3(name = "derivative")]
    #[pyo3(signature = (y, order, /, neighbors=3))]
    fn derivative_wrapper<'py>(
        py: Python<'py>,
        y: PyReadonlyArrayDyn<'py, f64>,
        order: usize,
        neighbors: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(derivative(y.as_slice()?, order, neighbors).into_pyarray(py))
    }
    /// Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
    /// by treating it as a periodic boundary
    ///
    /// Args:
    ///     data (np.ndarray): the data whose derivative should be taken
    ///     order (int): the order of the derivative
    ///     neighbors (int): the number of nearest neighbors to consider in each direction.
    /// Returns:
    ///     np.ndarray: the derivative
    #[pyfunction]
    #[pyo3(name = "derivative_periodic")]
    #[pyo3(signature = (y, order, /, neighbors=3))]
    fn derivative_periodic_wrapper<'py>(
        py: Python<'py>,
        y: PyReadonlyArrayDyn<'py, f64>,
        order: usize,
        neighbors: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(derivative_periodic(y.as_slice()?, order, neighbors).into_pyarray(py))
    }

    /// Reconstruct a measured frog trace.
    /// Args:
    ///     measurement_sg_sqrt: take the measurement, take its square root, and then fftshift along the frequency axis.
    ///     guess: provide a starting guess for the pulse
    ///     trial pulses: Number of different starting points to try (these will run in parallel)
    ///     iterations: Number of iterations to give each trial pulse before evaluating it against the others (only the best G' error is kept)
    ///     finishing_iteration: Number of additional iterations to apply to the result of the trial pulses which has the best G' error
    ///     frog_type (FrogType): The type of the frog as described in the FrogType enum
    ///     spectrum: optional spectral constraint. Should be interpolated onto the same grid as the spectrogram, and have fftshift applied.
    ///     measured_gate: optional gate pulse to use for xfrog
    ///     roi: region of interest array of booleans. true if in the region of interest. same size as the spectrgram frequency axis
    ///     ptycho_threshhold: value gamma of the threshholding operation suggested in the original ptychographic frog paper
    ///
    /// Returns:
    ///     pulse: the reconstructed pulse
    ///     gate: the reconstructed gate
    ///     error: the G' error of the reconstruction
    #[pyfunction]
    #[pyo3(name = "rust_frog")]
    #[pyo3(signature = (measurement_sg_sqrt, guess=None, trial_pulses=64, iterations=128, finishing_iterations=512, frog_type=FrogType::Shg, spectrum=None, measured_gate=None, roi=None, ptycho_threshhold=None))]
    fn frog_wrapper<'py>(
        py: Python<'py>,
        measurement_sg_sqrt: PyReadonlyArrayDyn<'py, f64>,
        guess: Option<PyReadonlyArrayDyn<'py, Complex64>>,
        trial_pulses: usize,
        iterations: usize,
        finishing_iterations: usize,
        frog_type: FrogType,
        spectrum: Option<PyReadonlyArrayDyn<'py, f64>>,
        measured_gate: Option<PyReadonlyArrayDyn<'py, Complex64>>,
        roi: Option<PyReadonlyArrayDyn<'py, bool>>,
        ptycho_threshhold: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<Complex64>>,
        Bound<'py, PyArray1<Complex64>>,
        f64,
        usize,
        usize,
    )> {
        let guess_option: Option<Vec<Complex64>> = match guess {
            Some(g) => Some(g.as_slice()?.to_vec()),
            None => None,
        };
        let spectrum_option: Option<Vec<f64>> = match spectrum {
            Some(s) => Some(s.as_slice()?.to_vec()),
            None => None,
        };
        let measured_gate_option: Option<Vec<Complex64>> = match measured_gate {
            Some(s) => Some(s.as_slice()?.to_vec()),
            None => None,
        };
        let roi_option: Option<Vec<bool>> = match roi {
            Some(s) => Some(s.as_slice()?.to_vec()),
            None => None,
        };
        let (pulse, gate, g_error, trial_best_index, finishing_best_index) = reconstruct_frog(
            measurement_sg_sqrt.as_slice()?,
            guess_option.as_ref().map(|vec| vec.as_slice()),
            trial_pulses,
            iterations,
            finishing_iterations,
            frog_type,
            spectrum_option.as_ref().map(|vec| vec.as_slice()),
            measured_gate_option.as_ref().map(|vec| vec.as_slice()),
            roi_option.as_ref().map(|vec| vec.as_slice()),
            ptycho_threshhold,
        );
        Ok((
            pulse.to_pyarray(py),
            gate.to_pyarray(py),
            g_error,
            trial_best_index,
            finishing_best_index,
        ))
    }
}
