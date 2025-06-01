use pyo3::prelude::*;

/// Functions written in Rust for improved performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_hello, m)?)?;
    m.add_function(wrap_pyfunction!(fornberg_stencil, m)?)?;
    Ok(())
}

/// Test function to make sure the Rust module is working
#[pyfunction]
fn rust_hello() -> PyResult<()> {
    println!("Hi from Rust!");
    Ok(())
}

#[pyfunction]
/// Generate a finite difference stencil using the algorithm described by B. Fornberg
/// in Mathematics of Computation 51, 699-706 (1988).
///
/// Args:
///     order (int): the order of the derivative
///     positions (np.ndarray): the positions at which the functions will be evaluated in the stencil. Must be larger than 2 elements in size.
/// Returns:
///     np.ndarray: the finite difference stencil with weights corresponding to the positions in the positions input array
///
/// Examples:
///
///     >>> stencil = fornberg_stencil(1, [-1,0,1])
///     >>> print(stencil)
///     [-0.5 0. 0.5]
fn fornberg_stencil(order: usize, positions: Vec<f64>, position_out: f64) -> Vec<f64> {
    fornberg_stencil_reference(order, &positions, position_out)
}

/// Internal version of fornberg_stencil() which takes positions by reference
fn fornberg_stencil_reference(order: usize, positions: &[f64], position_out: f64) -> Vec<f64> {
    let n_pos = positions.len();
    let mut delta_current = vec![vec![0.0; order + 1]; n_pos];
    let mut delta_previous = vec![vec![0.0; order + 1]; n_pos];
    delta_previous[0][0] = 1.0;

    let mut c1 = 1.0;
    for n in 1..n_pos {
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
        if n < (n_pos - 1) {
            delta_previous = delta_current.clone();
        }
    }

    (0..n_pos).map(|v| delta_current[v][order]).collect()
}
