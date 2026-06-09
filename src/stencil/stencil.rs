/// Internal version of fornberg_stencil() which takes positions by reference
pub fn fornberg_stencil(order: usize, positions: &[f64], position_out: f64) -> Box<[f64]> {
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

                delta_current[m * n_pos + v] =
                    ((positions[n] - position_out) * delta_previous[m * n_pos + v] - last_element)
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
