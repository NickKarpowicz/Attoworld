use rayon::prelude::*;
mod stencil;
pub use stencil::fornberg_stencil;
/// find the interpolated location and maximum value of a distribution
/// Returns:
///     (location , interpolated maximum) both as f64, where the digits after location
///     indicate where between the pixels the interpolated position sits
pub fn find_maximum_location(y: &[f64], neighbors: i64) -> Result<(f64, f64), ()> {
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
pub fn sort_paired_xy(x_in: &[f64], y_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
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

/// Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
/// by treating it as a periodic boundary
///
/// Args:
///     data: the data whose derivative should be taken
///     order: the order of the derivative
///     neighbors: the number of nearest neighbors to consider in each direction.
/// Returns:
///     the derivative
pub fn derivative(y: &[f64], order: usize, neighbors: usize) -> Box<[f64]> {
    let positions: Box<[f64]> = (0..(2 * neighbors + 1))
        .map(|a| a as f64 - neighbors as f64)
        .collect();
    let front_edge_positions: Box<[f64]> = (0..=(2 * neighbors + 2)).map(|a| a as f64).collect();
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
pub fn derivative_periodic(y: &[f64], order: usize, neighbors: usize) -> Box<[f64]> {
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
pub fn find_first_intercept(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
    find_first_intercept_core(y.iter(), y.len() - 1usize, intercept_value, neighbors)
}
/// find the last intercept between intercept_value and y, using a finite difference stencil defined by neighbors
pub fn find_last_intercept(y: &[f64], intercept_value: f64, neighbors: usize) -> f64 {
    let last_element_index = y.len() - 1usize;
    last_element_index as f64
        - find_first_intercept_core(
            y.iter().rev(),
            last_element_index,
            intercept_value,
            neighbors,
        )
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
