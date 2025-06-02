use std::f64;

use pyo3::prelude::*;

/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fornberg_stencil_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(find_maximum_location_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(find_first_intercept_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(find_last_intercept_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(fwhm, m)?)?;
    Ok(())
}

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
fn find_maximum_location_wrapper(y: Vec<f64>, neighbors: i64) -> (f64, f64) {
    find_maximum_location(&y, neighbors)
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
fn find_first_intercept_wrapper(y: Vec<f64>, intercept_value: f64, neighbors: usize) -> f64 {
    find_first_intercept(&y, intercept_value, neighbors)
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
fn find_last_intercept_wrapper(y: Vec<f64>, intercept_value: f64, neighbors: usize) -> f64 {
    find_last_intercept(&y, intercept_value, neighbors)
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
fn fwhm(y: Vec<f64>, dx: f64, intercept_value: f64, neighbors: usize) -> f64 {
    let (_, max_value) = find_maximum_location(&y, neighbors as i64);
    let first_intercept = find_first_intercept(&y, max_value * intercept_value, neighbors);
    let last_intercept = find_last_intercept(&y, max_value * intercept_value, neighbors);
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
#[pyfunction]
#[pyo3(name = "fornberg_stencil")]
#[pyo3(signature = (order, positions, position_out = 0.0, /))]
fn fornberg_stencil_wrapper(
    order: usize,
    positions: Vec<f64>,
    position_out: f64,
) -> PyResult<Vec<f64>> {
    Ok(fornberg_stencil(order, &positions, position_out))
}

/// Internal version of fornberg_stencil() which takes positions by reference
fn fornberg_stencil(order: usize, positions: &[f64], position_out: f64) -> Vec<f64> {
    let n_pos = positions.len();
    let mut delta_current = vec![vec![0.0; order + 1]; n_pos];
    let mut delta_previous = vec![vec![0.0; order + 1]; n_pos];
    delta_current[0][0] = 1.0;

    let mut c1 = 1.0;
    for n in 1..n_pos {
        std::mem::swap(&mut delta_previous, &mut delta_current);
        let mut c2 = 1.0;
        for v in 0..n {
            let c3 = positions[n] - positions[v];
            c2 *= c3;

            if n <= order {
                delta_previous[v][n] = 0.0;
            }

            let min_n_order = std::cmp::min(n, order);
            for m in 0..=min_n_order {
                let last_element = if m == 0 {
                    0.0
                } else {
                    m as f64 * delta_previous[v][m - 1]
                };

                delta_current[v][m] =
                    ((positions[n] - position_out) * delta_previous[v][m] - last_element) / c3;
            }
        }

        let min_n_order = std::cmp::min(n, order);
        for m in 0..=min_n_order {
            let first_element = if m == 0 {
                0.0
            } else {
                m as f64 * delta_previous[n - 1][m - 1]
            };

            delta_current[n][m] = (c1 / c2)
                * (first_element - (positions[n - 1] - position_out) * delta_previous[n - 1][m]);
        }

        c1 = c2;
    }

    (0..n_pos).map(|v| delta_current[v][order]).collect()
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
fn clamp_index(x0: usize, lower_bound: usize, upper_bound: usize) -> usize {
    let (lower, upper) = if lower_bound <= upper_bound {
        (lower_bound, upper_bound)
    } else {
        (upper_bound, lower_bound)
    };
    std::cmp::max(lower, std::cmp::min(x0, upper))
}

fn find_first_intercept_core<'a>(
    y_iter: impl Iterator<Item = &'a f64> + Clone,
    last_element_index: usize,
    intercept_value: f64,
    neighbors: usize,
) -> f64 {
    if let Some(intercept_index) = y_iter.clone().position(|x| *x >= intercept_value) {
        let range_start = clamp_index(
            intercept_index - neighbors,
            0usize,
            last_element_index - 2 * neighbors,
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
