use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::f64;
use rayon::prelude::*;


/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
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
        }
        else{
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
                        .binary_search_by(|a| a.partial_cmp(x).unwrap_or(std::cmp::Ordering::Greater))
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
                        .binary_search_by(|a| a.partial_cmp(x).unwrap_or(std::cmp::Ordering::Greater))
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
    Ok(())
}
