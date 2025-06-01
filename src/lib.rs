use pyo3::prelude::*;

/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_hello, m)?)?;
    m.add_function(wrap_pyfunction!(fornberg_stencil_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(find_maximum_location_wrapper, m)?)?;
    Ok(())
}

/// Test function to make sure the Rust module is working
#[pyfunction]
fn rust_hello() -> PyResult<()> {
    println!("Hi from Rust!");
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
    //println!("derivatives {:?}", derivatives);

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
