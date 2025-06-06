use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64;

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
    fn find_maximum_location_wrapper<'py>(
        y: PyReadonlyArrayDyn<'py, f64>,
        neighbors: i64,
    ) -> (f64, f64) {
        find_maximum_location(y.as_slice().unwrap(), neighbors)
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
    fn find_first_intercept_wrapper<'py>(
        y: PyReadonlyArrayDyn<'py, f64>,
        intercept_value: f64,
        neighbors: usize,
    ) -> f64 {
        find_first_intercept(y.as_slice().unwrap(), intercept_value, neighbors)
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
    fn find_last_intercept_wrapper<'py>(
        y: PyReadonlyArrayDyn<'py, f64>,
        intercept_value: f64,
        neighbors: usize,
    ) -> f64 {
        find_last_intercept(y.as_slice().unwrap(), intercept_value, neighbors)
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
    fn fwhm<'py>(
        y: PyReadonlyArrayDyn<'py, f64>,
        dx: f64,
        intercept_value: f64,
        neighbors: usize,
    ) -> f64 {
        let (_, max_value) = find_maximum_location(y.as_slice().unwrap(), neighbors as i64);
        let first_intercept = find_first_intercept(
            y.as_slice().unwrap(),
            max_value * intercept_value,
            neighbors,
        );
        let last_intercept = find_last_intercept(
            y.as_slice().unwrap(),
            max_value * intercept_value,
            neighbors,
        );
        dx * (last_intercept - first_intercept)
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
    ///     >>> stencil = fornberg_stencil(1, [-1,0,1])
    ///     >>> print(stencil)
    ///     [-0.5 0. 0.5]
    #[pyfn(m)]
    #[pyo3(name = "fornberg_stencil")]
    #[pyo3(signature = (order, positions, position_out = 0.0, /))]
    fn fornberg_stencil_wrapper<'py>(
        py: Python<'py>,
        order: usize,
        positions: PyReadonlyArrayDyn<'py, f64>,
        position_out: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        fornberg_stencil(order, positions.as_slice().unwrap(), position_out).into_pyarray(py)
    }

    /// Internal version of fornberg_stencil() which takes positions by reference
    fn fornberg_stencil(order: usize, positions: &[f64], position_out: f64) -> Vec<f64> {
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
        delta_current[order * n_pos..(order + 1) * n_pos].to_vec()
    }

    fn find_maximum_location(y: &[f64], neighbors: i64) -> (f64, f64) {
        let max_index: i64 = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
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

        let stencil_positions: Vec<f64> = (start_index..(start_index + (2 * neighbors) as usize))
            .map(|x| x as f64)
            .collect();
        let mut derivatives: Vec<f64> = vec![0.0; (2 * neighbors) as usize + 1usize];
        for n in 0usize..=((2 * neighbors) as usize) {
            let stencil = fornberg_stencil(
                1usize,
                &stencil_positions,
                (max_index - 1) as f64 + (n as f64) / (neighbors as f64),
            );
            derivatives[n] = stencil
                .iter()
                .zip(y[start_index..(start_index + 2 * neighbors as usize)].iter())
                .map(|(x, y)| x * y)
                .sum();
        }

        let zero_xing_positions: Vec<f64> = (0..=(2 * neighbors))
            .map(|x| (max_index - 1) as f64 + (x as f64) / (neighbors as f64))
            .collect();
        let zero_xing_stencil = fornberg_stencil(0, &derivatives, 0.0);

        let location: f64 = zero_xing_stencil
            .iter()
            .zip(zero_xing_positions.iter())
            .map(|(x, y)| x * y)
            .sum();

        let interpolation_stencil = fornberg_stencil(0usize, &stencil_positions, location);

        let interpolated_max = interpolation_stencil
            .iter()
            .zip(y[start_index..(start_index + 2 * neighbors as usize)].iter())
            .map(|(x, y)| x * y)
            .sum();

        (location, interpolated_max)
    }

    fn sort_paired_xy(x_in: &[f64], y_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pairs: Vec<(f64, f64)> = x_in
            .iter()
            .zip(y_in.iter())
            .map(|(a, b)| (*a, *b))
            .collect();

        pairs.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
    ) -> Bound<'py, PyArray1<f64>> {
        if inputs_are_sorted {
            interpolate_sorted_1d_slice(
                x_out.as_slice().unwrap(),
                x_in.as_slice().unwrap(),
                y_in.as_slice().unwrap(),
                neighbors,
                extrapolate,
                derivative_order,
            )
            .into_pyarray(py)
        } else {
            let (x_in_sorted, y_in_sorted) =
                sort_paired_xy(x_in.as_slice().unwrap(), y_in.as_slice().unwrap());
            interpolate_sorted_1d_slice(
                x_out.as_slice().unwrap(),
                &x_in_sorted,
                &y_in_sorted,
                neighbors,
                extrapolate,
                derivative_order,
            )
            .into_pyarray(py)
        }
    }

    fn interpolate_sorted_1d_slice(
        x_out: &[f64],
        x_in: &[f64],
        y_in: &[f64],
        neighbors: i64,
        extrapolate: bool,
        derivative_order: usize,
    ) -> Vec<f64> {
        let core_stencil_size: usize = 2 * neighbors as usize;
        x_out
            .par_iter()
            .map(|x| {
                let index: usize = match x_in.binary_search_by(|a| a.partial_cmp(&x).unwrap()) {
                    Ok(index) => index,
                    Err(index) => index,
                };
                if (index == 0 || index == x_in.len()) && !extrapolate {
                    0.0
                } else {
                    let clamped_index: usize =
                        clamp_index(index as i64, neighbors, x_in.len() as i64 - neighbors)
                            - neighbors as usize;
                    let stencil_size: usize = if clamped_index == 0
                        || clamped_index == x_in.len() - (core_stencil_size - 1) as usize
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

    ///     Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
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
    ) -> Bound<'py, PyArray1<f64>> {
        derivative(y.as_slice().unwrap(), order, neighbors).into_pyarray(py)
    }
    ///     Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
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
    ) -> Bound<'py, PyArray1<f64>> {
        derivative_periodic(y.as_slice().unwrap(), order, neighbors).into_pyarray(py)
    }

    fn derivative(y: &[f64], order: usize, neighbors: usize) -> Vec<f64> {
        let positions: Vec<f64> = (0..(2 * neighbors + 1))
            .map(|a| a as f64 - neighbors as f64)
            .collect();
        let front_edge_positions: Vec<f64> = (0..=(2 * neighbors + 2)).map(|a| a as f64).collect();
        let rear_edge_positions: Vec<f64> = front_edge_positions
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

    fn derivative_periodic(y: &[f64], order: usize, neighbors: usize) -> Vec<f64> {
        let positions: Vec<f64> = (0..(2 * neighbors + 1))
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

    fn clamp_index(x0: i64, lower_bound: i64, upper_bound: i64) -> usize {
        let (lower, upper) = if lower_bound <= upper_bound {
            (lower_bound, upper_bound)
        } else {
            (upper_bound, lower_bound)
        };
        std::cmp::max(lower, std::cmp::min(x0, upper)) as usize
    }

    fn find_first_intercept_core<'a>(
        y_iter: impl Iterator<Item = &'a f64> + Clone,
        last_element_index: usize,
        intercept_value: f64,
        neighbors: usize,
    ) -> f64 {
        if let Some(intercept_index) = y_iter.clone().position(|x| *x >= intercept_value) {
            let range_start = clamp_index(
                intercept_index as i64 - neighbors as i64,
                0,
                last_element_index as i64 - 2 * neighbors as i64,
            );
            let range_i: Vec<usize> = y_iter
                .clone()
                .enumerate()
                .skip(range_start)
                .take(2 * neighbors)
                .scan((None, None), |state, (index, value)| {
                    if state.0.is_none() || *value > state.1.unwrap() {
                        state.0 = Some(index);
                        state.1 = Some(*value);
                        Some(Some(index))
                    } else {
                        state.0 = Some(index);
                        state.1 = Some(*value);
                        Some(None)
                    }
                })
                .flatten()
                .collect();

            let x_positions: Vec<f64> = range_i.iter().map(|x| *x as f64).collect();
            let y_values: Vec<f64> = y_iter
                .enumerate()
                .skip(range_start)
                .take(2 * neighbors)
                .filter_map(|(index, value)| range_i.contains(&index).then(|| *value))
                .collect();
            let stencil = fornberg_stencil(0, &y_values, intercept_value);
            stencil
                .iter()
                .zip(x_positions.iter())
                .map(|(a, b)| a * b)
                .sum()
        } else {
            f64::NAN
        }
    }

    fn find_first_intercept(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
        find_first_intercept_core(y.iter(), y.len() - 1usize, intercept_value, neighbors)
    }

    fn find_last_intercept<'a>(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
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
