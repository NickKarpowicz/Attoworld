use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex};
use std::{f64, thread};

/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyclass]
    #[derive(Clone)]
    enum FrogType {
        Shg,
        Thg,
        Kerr,
        Xfrog,
        Blindfrog,
    }
    m.add_class::<FrogType>()?;

    /// Find the location and value of the maximum of a smooth, uniformly sampled signal, interpolating to find the sub-pixel location
    ///
    /// Args:
    ///     y (np.ndarray): The signal whose maximum should be located
    ///     neighbors (int): the number of neighboring points to consider in the optimization (default 3)
    ///
    /// Returns:
    ///     (float, float): location, interpolated maximum
    #[pyfn(m)]
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
    #[pyfn(m)]
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
    #[pyfn(m)]
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
    #[pyfn(m)]
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
                ))
            }
        };
        let first_intercept =
            find_first_intercept(y.as_slice()?, max_value * intercept_value, neighbors);
        let last_intercept =
            find_last_intercept(y.as_slice()?, max_value * intercept_value, neighbors);
        if first_intercept > last_intercept {
            println!("Warning: internal calculation give a negative width, data may be too coarse to be reliable.");
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
    #[pyfn(m)]
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

    /// Internal version of fornberg_stencil() which takes positions by reference
    fn fornberg_stencil(order: usize, positions: &[f64], position_out: f64) -> Box<[f64]> {
        let n_pos = positions.len();
        let cols = order + 1;
        let mut delta_current = vec![0.0; n_pos * cols];
        let mut delta_previous = vec![0.0; n_pos * cols];
        delta_current[0] = 1.0;
        let mut c1 = 1.0;
        for n in 1..n_pos {
            std::mem::swap(&mut delta_previous, &mut delta_current);
            let mut c2 = 1.0;
            let zero_previous = n <= order;
            let min_n_order = std::cmp::min(n, order);
            for v in 0..n {
                let c3 = positions[n] - positions[v];
                c2 *= c3;

                if zero_previous {
                    delta_previous[n * n_pos + v] = 0.0;
                }

                for m in 0..=min_n_order {
                    let last_element = if m == 0 {
                        0.0
                    } else {
                        m as f64 * delta_previous[(m - 1) * n_pos + v]
                    };

                    delta_current[m * n_pos + v] = ((positions[n] - position_out)
                        * delta_previous[m * n_pos + v]
                        - last_element)
                        / c3;
                }
            }

            for m in 0..=min_n_order {
                let first_element = if m == 0 {
                    0.0
                } else {
                    m as f64 * delta_previous[(m - 1) * n_pos + n - 1]
                };

                delta_current[m * n_pos + n] = (c1 / c2)
                    * (first_element
                        - (positions[n - 1] - position_out) * delta_previous[m * n_pos + n - 1]);
            }

            c1 = c2;
        }
        delta_current[order * n_pos..cols * n_pos].into()
    }

    /// find the interpolated location and maximum value of a distribution
    /// Returns:
    ///     (location , interpolated maximum) both as f64, where the digits after location
    ///     indicate where between the pixels the interpolated position sits
    fn find_maximum_location(y: &[f64], neighbors: i64) -> Result<(f64, f64), ()> {
        let max_index: i64 = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
            .ok_or(())?
            .0 as i64;

        let start_index: usize =
            if (max_index - neighbors) >= 0 && (max_index + neighbors) < y.len() as i64 {
                if y[(max_index + 1) as usize] > y[(max_index - 1) as usize] {
                    (max_index - neighbors + 1) as usize
                } else {
                    (max_index - neighbors) as usize
                }
            } else if (max_index - neighbors) < 0 {
                0usize
            } else {
                y.len() - 2 * neighbors as usize - 1usize
            };

        let stencil_positions: Box<[f64]> = (start_index..(start_index + (2 * neighbors) as usize))
            .map(|x| x as f64)
            .collect();

        let derivatives: Box<[f64]> = (0usize..=((2 * neighbors) as usize))
            .map(|n| {
                fornberg_stencil(
                    1usize,
                    &stencil_positions,
                    (max_index - 1) as f64 + (n as f64) / (neighbors as f64),
                )
                .iter()
                .zip(y[start_index..(start_index + 2 * neighbors as usize)].iter())
                .map(|(x, y)| x * y)
                .sum()
            })
            .collect();

        let zero_xing_positions: Box<[f64]> = (0..=(2 * neighbors))
            .map(|x| (max_index - 1) as f64 + (x as f64) / (neighbors as f64))
            .collect();

        let location: f64 = fornberg_stencil(0, &derivatives, 0.0)
            .iter()
            .zip(zero_xing_positions.iter())
            .map(|(x, y)| x * y)
            .sum();

        let interpolated_max = fornberg_stencil(0usize, &stencil_positions, location)
            .iter()
            .zip(y[start_index..(start_index + 2 * neighbors as usize)].iter())
            .map(|(x, y)| x * y)
            .sum();

        Ok((location, interpolated_max))
    }

    /// Sort x,y values in two slices such that x values are in ascending order

    fn sort_paired_xy(x_in: &[f64], y_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pairs: Vec<(f64, f64)> = x_in
            .iter()
            .zip(y_in.iter())
            .map(|(a, b)| (*a, *b))
            .collect();

        if cfg!(target_arch = "wasm32") {
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater));
        } else {
            pairs.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater));
        }

        pairs.into_iter().unzip()
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
    #[pyfn(m)]
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
    /// Interpolate sorted data, given a list of intersection locations
    ///
    /// Args:
    ///     x_out: array of output x values, the array onto which y_in will be interpolated
    ///     x_in: array of input x values
    ///     y_in: array of input y values
    ///     neighbors: number of nearest neighbors to include in the interpolation
    ///     extrapolate: unless set to true, values outside of the range of x_in will be zero
    ///     derivative_order: order of derivative to take. 0 (default) is plain interpolation, 1 takes first derivative, and so on.
    ///
    /// Returns:
    ///     the interpolated y_out
    pub fn interpolate_sorted_1d_slice(
        x_out: &[f64],
        x_in: &[f64],
        y_in: &[f64],
        neighbors: i64,
        extrapolate: bool,
        derivative_order: usize,
    ) -> Box<[f64]> {
        let core_stencil_size: usize = 2 * neighbors as usize;

        //note that the only difference here is the use of .iter() or .par_iter() at the beginning of the chain.
        if cfg!(target_arch = "wasm32") {
            x_out
                .iter()
                .map(|x| {
                    let index: usize = x_in
                        .binary_search_by(|a| {
                            a.partial_cmp(x).unwrap_or(std::cmp::Ordering::Greater)
                        })
                        .unwrap_or_else(|e| e);
                    if (index == 0 || index == x_in.len()) && !extrapolate {
                        0.0
                    } else {
                        let clamped_index: usize = index
                            .clamp(neighbors as usize, (x_in.len() as i64 - neighbors) as usize)
                            - neighbors as usize;
                        let stencil_size: usize = if clamped_index == 0
                            || clamped_index == x_in.len() - (core_stencil_size - 1)
                        {
                            core_stencil_size + 1
                        } else {
                            core_stencil_size
                        };
                        //finite difference stencil with order 0 is interpolation
                        fornberg_stencil(
                            derivative_order,
                            &x_in[clamped_index..(clamped_index + stencil_size)],
                            *x,
                        )
                        .iter()
                        .zip(y_in.iter().skip(clamped_index))
                        .map(|(a, b)| a * b)
                        .sum()
                    }
                })
                .collect()
        } else {
            x_out
                .par_iter()
                .map(|x| {
                    let index: usize = x_in
                        .binary_search_by(|a| {
                            a.partial_cmp(x).unwrap_or(std::cmp::Ordering::Greater)
                        })
                        .unwrap_or_else(|e| e);
                    if (index == 0 || index == x_in.len()) && !extrapolate {
                        0.0
                    } else {
                        let clamped_index: usize = index
                            .clamp(neighbors as usize, (x_in.len() as i64 - neighbors) as usize)
                            - neighbors as usize;
                        let stencil_size: usize = if clamped_index == 0
                            || clamped_index == x_in.len() - (core_stencil_size - 1)
                        {
                            core_stencil_size + 1
                        } else {
                            core_stencil_size
                        };
                        //finite difference stencil with order 0 is interpolation
                        fornberg_stencil(
                            derivative_order,
                            &x_in[clamped_index..(clamped_index + stencil_size)],
                            *x,
                        )
                        .iter()
                        .zip(y_in.iter().skip(clamped_index))
                        .map(|(a, b)| a * b)
                        .sum()
                    }
                })
                .collect()
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
    #[pyfn(m)]
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
    #[pyfn(m)]
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
    /// Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
    /// by treating it as a periodic boundary
    ///
    /// Args:
    ///     data: the data whose derivative should be taken
    ///     order: the order of the derivative
    ///     neighbors: the number of nearest neighbors to consider in each direction.
    /// Returns:
    ///     the derivative
    fn derivative(y: &[f64], order: usize, neighbors: usize) -> Box<[f64]> {
        let positions: Box<[f64]> = (0..(2 * neighbors + 1))
            .map(|a| a as f64 - neighbors as f64)
            .collect();
        let front_edge_positions: Box<[f64]> =
            (0..=(2 * neighbors + 2)).map(|a| a as f64).collect();
        let rear_edge_positions: Box<[f64]> = front_edge_positions
            .iter()
            .map(|a| a + (y.len() - 2 * neighbors - 3) as f64)
            .collect();
        let inner_stencil = fornberg_stencil(order, &positions, 0.0);
        (0..y.len())
            .map(|index| {
                if index < neighbors {
                    let stencil = fornberg_stencil(order, &front_edge_positions, index as f64);
                    stencil
                        .iter()
                        .zip(y.iter())
                        .map(|(stencil_val, y_val)| *stencil_val * (*y_val))
                        .sum()
                } else if index > y.len() - neighbors - 1 {
                    let stencil = fornberg_stencil(order, &rear_edge_positions, index as f64);
                    stencil
                        .iter()
                        .zip(y.iter().skip(y.len() - 2 * neighbors - 3))
                        .map(|(stencil_val, y_val)| *stencil_val * *y_val)
                        .sum()
                } else {
                    y[index - neighbors..index + neighbors + 1]
                        .iter()
                        .zip(inner_stencil.iter())
                        .map(|(stencil_val, y_val)| *stencil_val * *y_val)
                        .sum()
                }
            })
            .collect()
    }

    /// similar to derivative() but the boundary conditions are periodic
    fn derivative_periodic(y: &[f64], order: usize, neighbors: usize) -> Box<[f64]> {
        let positions: Box<[f64]> = (0..(2 * neighbors + 1))
            .map(|a| a as f64 - neighbors as f64)
            .collect();
        let stencil = fornberg_stencil(order, &positions, 0.0);
        (0..y.len())
            .map(|index| {
                stencil
                    .iter()
                    .zip(y.iter().cycle().skip(y.len() - neighbors + index))
                    .map(|(a, b)| *a * *b)
                    .sum()
            })
            .collect()
    }

    /// finds the first intercept between the values contained in y_iter and intercept_value. last_element_index provides the index
    /// of the last element of the iter.
    /// neighbors specifies the number of nearest neighbors in each direction to use for finite difference stencils.
    fn find_first_intercept_core<'a>(
        y_iter: impl Iterator<Item = &'a f64> + Clone,
        last_element_index: usize,
        intercept_value: f64,
        neighbors: usize,
    ) -> f64 {
        if let Some(intercept_index) = y_iter.clone().position(|x| *x >= intercept_value) {
            let range_start = (intercept_index as i64 - neighbors as i64)
                .clamp(0, last_element_index as i64 - 2 * neighbors as i64)
                as usize;
            let range_i: Vec<usize> = y_iter
                .clone()
                .enumerate()
                .skip(range_start)
                .take(2 * neighbors)
                .scan((None, None), |state, (index, value)| match state.0 {
                    Some(_) => match state.1 {
                        Some(v) => {
                            if *value > v {
                                state.0 = Some(index);
                                state.1 = Some(*value);
                                Some(Some(index))
                            } else {
                                state.0 = Some(index);
                                state.1 = Some(*value);
                                Some(None)
                            }
                        }
                        None => Some(Some(index)),
                    },
                    None => {
                        state.0 = Some(index);
                        state.1 = Some(*value);
                        Some(Some(index))
                    }
                })
                .flatten()
                .collect();

            let x_positions: Box<[f64]> = range_i.iter().map(|x| *x as f64).collect();
            let y_values: Box<[f64]> = y_iter
                .enumerate()
                .skip(range_start)
                .take(2 * neighbors)
                .filter_map(|(index, value)| range_i.contains(&index).then_some(*value))
                .collect();
            fornberg_stencil(0, &y_values, intercept_value)
                .iter()
                .zip(x_positions.iter())
                .map(|(a, b)| a * b)
                .sum()
        } else {
            f64::NAN
        }
    }

    /// find the first intercept between intercept_value and y, using a finite difference stencil defined by neighbors
    fn find_first_intercept(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
        find_first_intercept_core(y.iter(), y.len() - 1usize, intercept_value, neighbors)
    }
    /// find the last intercept between intercept_value and y, using a finite difference stencil defined by neighbors
    fn find_last_intercept(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
        let last_element_index = y.len() - 1usize;
        last_element_index as f64
            - find_first_intercept_core(
                y.iter().rev(),
                last_element_index,
                intercept_value,
                neighbors,
            )
    }

    fn frog_guess_from_pulse_and_gate(
        pulse: &[Complex64],
        gate: &[Complex64],
        nonlinearity: FrogType,
    ) -> Vec<Complex64> {
        match nonlinearity {
            FrogType::Shg => pulse
                .iter()
                .zip(gate.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            _ => pulse.to_vec(),
        }
    }

    fn calculate_g_error(
        measurement_normalized: &[f64],
        pulse: &[Complex64],
        gate: &[Complex64],
        workspace: &mut [Complex64],
        reconstructed_spectrogram: &mut [f64],
        fft_forward: Arc<dyn Fft<f64>>,
    ) -> f64 {
        let mut norm_recon = 0.0;
        let dim = pulse.len();
        let dim_i: i64 = dim as i64;
        let half: i64 = dim as i64 / 2;
        for j in 0..dim {
            for i in 0..dim {
                let g_index: i64 = j as i64 - half + i as i64;
                if (g_index >= 0) && (g_index < dim_i) {
                    workspace[i] = pulse[i] * gate[g_index as usize];
                } else {
                    workspace[i] = Complex64::ZERO;
                }
            }
            fft_forward.process(workspace);
            for (a, b) in reconstructed_spectrogram
                .iter_mut()
                .skip(j * dim)
                .take(dim)
                .zip(workspace.iter())
            {
                *a = (b.conj() * *b).re;
                norm_recon += a.powi(2);
            }
        }
        norm_recon = norm_recon.sqrt();

        let mut variance = 0.0;
        for j in 0..dim {
            for i in 0..dim {
                variance += (measurement_normalized[i * dim + j]
                    - reconstructed_spectrogram[j * dim + i] / norm_recon)
                    .powi(2);
            }
        }
        let area = measurement_normalized.iter().map(|&a| a * a).sum::<f64>();
        return (variance / area).sqrt();
    }

    fn gate_from_pulse(
        field: &[Complex64],
        gate: &mut [Complex64],
        nonlinearity: &FrogType,
        measured_gate: Option<&[Complex64]>,
    ) {
        match nonlinearity {
            FrogType::Shg => {
                for (a, b) in gate.iter_mut().zip(field.iter()) {
                    *a = *b;
                }
            }
            FrogType::Thg => {
                for (a, b) in gate.iter_mut().zip(field.iter()) {
                    *a = *b * *b;
                }
            }
            FrogType::Kerr => {
                for (a, b) in gate.iter_mut().zip(field.iter()) {
                    *a = *b * b.conj();
                }
            }
            FrogType::Xfrog => {
                for (a, b) in gate.iter_mut().zip(measured_gate.unwrap().iter()) {
                    *a = *b;
                }
            }
            FrogType::Blindfrog => {}
        }
    }

    fn frog_apply_spectral_constraint(
        field: &mut [Complex64],
        spectrum: Option<&[f64]>,
        fft_forward: Arc<dyn Fft<f64>>,
        fft_backward: Arc<dyn Fft<f64>>,
    ) {
        match spectrum {
            Some(spec) => {
                fft_forward.process(field);
                for (a, b) in field.iter_mut().zip(spec.iter()) {
                    *a = Complex64::from_polar(*b, a.arg());
                }
                fft_backward.process(field);
            }
            None => return,
        }
    }

    fn get_norm_meas(meas_sqrt: &[f64]) -> Vec<f64> {
        let norm: f64 = meas_sqrt.iter().map(|&a| a.powi(4)).sum::<f64>().sqrt();
        meas_sqrt.iter().map(|&a| a * a / norm).collect()
    }

    fn generate_random_pulse(dim: usize) -> Vec<Complex64> {
        let range = rand::distr::Uniform::new(-1.0f64, 1.0f64).unwrap();
        let mut rng = rand::rng();
        (0..dim)
            .map(|_| Complex64::new(range.sample(&mut rng), range.sample(&mut rng)))
            .collect()
    }

    #[pyfn(m)]
    #[pyo3(name = "rust_frog")]
    #[pyo3(signature = (measurement_sg_sqrt, guess=None, trial_pulses=64, iterations=128, finishing_iterations=512, frog_type=FrogType::Shg, spectrum=None, measured_gate=None))]
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
    ) -> PyResult<(
        Bound<'py, PyArray1<Complex64>>,
        Bound<'py, PyArray1<Complex64>>,
        f64,
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
        let (pulse, gate, g_error) = reconstruct_frog(
            measurement_sg_sqrt.as_slice()?,
            guess_option.as_ref().map(|vec| vec.as_slice()),
            trial_pulses,
            iterations,
            finishing_iterations,
            frog_type,
            spectrum_option.as_ref().map(|vec| vec.as_slice()),
            measured_gate_option.as_ref().map(|vec| vec.as_slice()),
        );
        Ok((pulse.to_pyarray(py), gate.to_pyarray(py), g_error))
    }

    #[derive(Clone)]
    struct FrogResult {
        pulse: Vec<Complex64>,
        gate: Vec<Complex64>,
        error: f64,
    }
    impl FrogResult {
        fn swap_if_better(&mut self, other: FrogResult) {
            if other.error < self.error {
                *self = other;
            }
        }
    }

    fn reconstruct_frog(
        measurement_sg_sqrt: &[f64],
        guess: Option<&[Complex64]>,
        trial_pulses: usize,
        iterations: usize,
        finishing_iterations: usize,
        frog_type: FrogType,
        spectrum: Option<&[f64]>,
        measured_gate: Option<&[Complex64]>,
    ) -> (Vec<Complex64>, Vec<Complex64>, f64) {
        let mut alloc = FrogAllocation::new(
            measurement_sg_sqrt,
            guess.map(|x| x.to_vec()),
            None,
            frog_type.clone(),
            spectrum.map(|x| x.to_vec()),
            measured_gate.map(|x| x.to_vec()),
        );
        let best_result = Arc::new(Mutex::new(reconstruct_frog_core(alloc.clone(), iterations)));
        let threads: usize = thread::available_parallelism()
            .unwrap_or(core::num::NonZeroUsize::MIN)
            .get();
        let thread_pulses = (trial_pulses + threads - 1) / threads;
        let mut handles = Vec::with_capacity(threads);
        if cfg!(target_arch = "wasm32") {
            for _ in 0..trial_pulses {
                let new_result = reconstruct_frog_core(alloc.clone(), iterations);
                let mut best_result_lock = best_result.lock().unwrap();
                best_result_lock.swap_if_better(new_result);
            }
        } else {
            for _ in 0..threads {
                let best_result_clone = Arc::clone(&best_result);
                let local_alloc = alloc.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..thread_pulses {
                        let new_result = reconstruct_frog_core(local_alloc.clone(), iterations);
                        let mut best_result_lock = best_result_clone.lock().unwrap();
                        best_result_lock.swap_if_better(new_result);
                    }
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }
        }

        let result = best_result.lock().unwrap();
        alloc.guess = Some(result.pulse.clone());
        alloc.gate_guess = Some(result.gate.clone());
        let last = reconstruct_frog_core(alloc, finishing_iterations);
        (last.pulse, last.gate, last.error)
    }

    #[derive(Clone)]
    struct FrogAllocation {
        measurement_sg_sqrt: Vec<f64>,
        measurement_normalized: Vec<f64>,
        guess: Option<Vec<Complex64>>,
        gate_guess: Option<Vec<Complex64>>,
        dim: usize,
        frog_type: FrogType,
        spectrum: Option<Vec<f64>>,
        measured_gate: Option<Vec<Complex64>>,
        fft_forward: Arc<dyn Fft<f64>>,
        fft_backward: Arc<dyn Fft<f64>>,
        workspace: Vec<Complex64>,
        reconstructed_spectrogram: Vec<f64>,
    }
    impl FrogAllocation {
        fn new(
            measurement_sg_sqrt: &[f64],
            guess: Option<Vec<Complex64>>,
            gate_guess: Option<Vec<Complex64>>,
            frog_type: FrogType,
            spectrum: Option<Vec<f64>>,
            measured_gate: Option<Vec<Complex64>>,
        ) -> Self {
            let dim: usize = ((measurement_sg_sqrt.len() as f64).sqrt()).round() as usize;
            let measurement_sg_sqrt = measurement_sg_sqrt.to_vec();
            let workspace = vec![Complex64::new(0.0, 0.0); dim];
            let reconstructed_spectrogram = vec![0.0f64; dim * dim];
            let mut planner = FftPlanner::<f64>::new();
            let fft_forward = planner.plan_fft_forward(dim);
            let fft_backward = planner.plan_fft_inverse(dim);
            let measurement_normalized = get_norm_meas(&measurement_sg_sqrt);
            FrogAllocation {
                measurement_sg_sqrt,
                measurement_normalized,
                guess,
                gate_guess,
                dim,
                frog_type,
                spectrum,
                measured_gate,
                fft_forward,
                fft_backward,
                workspace,
                reconstructed_spectrogram,
            }
        }
    }

    fn reconstruct_frog_core(mut alloc: FrogAllocation, iterations: usize) -> FrogResult {
        let mut pulse = match alloc.guess {
            Some(field) => field,
            None => generate_random_pulse(alloc.dim),
        };

        let mut gate: Vec<Complex64> = match alloc.gate_guess {
            Some(g) => g,
            None => {
                let mut g = pulse.clone();
                gate_from_pulse(
                    &pulse,
                    &mut g,
                    &alloc.frog_type,
                    alloc.measured_gate.as_deref(),
                );
                g
            }
        };

        let mut best = pulse.clone();
        let mut best_gate = gate.clone();
        let mut best_error: f64 = calculate_g_error(
            &alloc.measurement_normalized,
            &pulse,
            &gate,
            &mut alloc.workspace,
            &mut alloc.reconstructed_spectrogram,
            alloc.fft_forward.clone(),
        );

        for _ in 0..iterations {
            (pulse, gate) = apply_frog_iteration(
                &pulse,
                &gate,
                &mut alloc.workspace,
                alloc.measurement_sg_sqrt.as_slice(),
                alloc.fft_forward.clone(),
                alloc.fft_backward.clone(),
            );
            pulse = frog_guess_from_pulse_and_gate(&pulse, &gate, FrogType::Shg);
            frog_apply_spectral_constraint(
                &mut pulse,
                alloc.spectrum.as_deref(),
                alloc.fft_forward.clone(),
                alloc.fft_backward.clone(),
            );
            gate_from_pulse(
                &pulse,
                &mut gate,
                &alloc.frog_type,
                alloc.measured_gate.as_deref(),
            );
            let g_error = calculate_g_error(
                alloc.measurement_normalized.as_slice(),
                &pulse,
                &gate,
                &mut alloc.workspace,
                &mut alloc.reconstructed_spectrogram,
                alloc.fft_forward.clone(),
            );

            if g_error < best_error {
                best_error = g_error;
                best_gate = gate.clone();
                best = pulse.clone();
            }
        }

        return FrogResult {
            pulse: best,
            gate: best_gate,
            error: best_error,
        };
    }

    #[pyfn(m)]
    #[pyo3(name = "frog_iteration")]
    #[pyo3(signature = (input_field, input_gate, meas_sqrt))]
    fn frog_iteration_wrapper<'py>(
        py: Python<'py>,
        input_field: PyReadonlyArrayDyn<'py, Complex64>,
        input_gate: PyReadonlyArrayDyn<'py, Complex64>,
        meas_sqrt: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<Complex64>>,
        Bound<'py, PyArray1<Complex64>>,
    )> {
        let dim = input_field.as_slice()?.len();
        let mut workspace = vec![Complex64::new(0.0, 0.0); dim];
        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(dim);
        let fft_backward = planner.plan_fft_inverse(dim);
        let (field, gate) = apply_frog_iteration(
            input_field.as_slice()?,
            input_gate.as_slice()?,
            &mut workspace,
            meas_sqrt.as_slice()?,
            fft_forward.clone(),
            fft_backward,
        );

        Ok((field.into_pyarray(py), gate.into_pyarray(py)))
    }

    fn apply_frog_iteration(
        input_field: &[Complex64],
        input_gate: &[Complex64],
        workspace: &mut [Complex64],
        meas_sqrt: &[f64],
        fft_forward: Arc<dyn Fft<f64>>,
        fft_backward: Arc<dyn Fft<f64>>,
    ) -> (Vec<Complex64>, Vec<Complex64>) {
        let dim: usize = input_field.len();
        let dim_i: i64 = dim as i64;
        let half: i64 = dim_i / 2;
        let mut field = vec![Complex64::ZERO; dim];
        let mut gate = vec![Complex64::ZERO; dim];
        workspace.fill(Complex64::ZERO);

        for j in 0..dim {
            for i in 0..dim {
                let g_index: i64 = j as i64 - half + i as i64;
                if (g_index >= 0) && (g_index < dim_i) {
                    workspace[i] = input_field[i] * input_gate[g_index as usize];
                } else {
                    workspace[i] = Complex64::ZERO;
                }
            }
            fft_forward.process(workspace);
            for i in 0..dim {
                workspace[i] =
                    Complex64::from_polar(meas_sqrt[i * dim + j], workspace[i].clone().arg());
            }
            fft_backward.process(workspace);
            for i in 0..dim {
                field[i] += workspace[i];
                let g_index: i64 = j as i64 - half + i as i64;
                if (g_index >= 0) && (g_index < dim_i) {
                    gate[g_index as usize] += workspace[i]
                }
            }
        }
        (field, gate)
    }
    Ok(())
}
