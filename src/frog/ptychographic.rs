use rand::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::Fft;
use std::sync::Arc;
fn ptychographic_threshold(x: Complex64, gamma: f64) -> Complex64 {
    let real: f64 = if x.re.abs() < gamma {
        0.0
    } else {
        x.re - x.re.signum() * gamma
    };

    let imag: f64 = if x.im.abs() < gamma {
        0.0
    } else {
        x.im - x.im.signum() * gamma
    };
    Complex64::new(real, imag)
}

pub fn apply_ptychographic_frog_iteration(
    input_field: &[Complex64],
    meas_sqrt: &[f64],
    workspace: &mut [Complex64],
    fft_forward: Arc<dyn Fft<f64>>,
    fft_backward: Arc<dyn Fft<f64>>,
    roi: &[bool],
    threshhold_gamma: f64,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let dim: usize = input_field.len();
    let dim_i: i64 = dim as i64;
    let half = dim_i / 2;
    let mut field = input_field.to_vec();
    let mut rng = rand::rng();
    let alpha_range = rand::distr::Uniform::new(0.1f64, 0.5f64).unwrap();
    let mut randomized_indices: Vec<usize> = (0usize..dim).collect::<Vec<usize>>();
    randomized_indices.shuffle(&mut rng);
    for j in randomized_indices {
        let alpha = alpha_range.sample(&mut rng);
        let field_max_squared: f64 = field
            .iter()
            .map(|&x| x.re * x.re + x.im * x.im)
            .reduce(f64::max)
            .unwrap_or(f64::MAX);

        for i in 0..dim {
            let g_index: i64 = j as i64 - half + i as i64;
            if (g_index >= 0) && (g_index < dim_i) {
                workspace[i] = field[g_index as usize];
            } else {
                workspace[i] = Complex64::ZERO;
            }
        }
        let mut nonlinear_product: Vec<Complex64> = field
            .iter()
            .zip(workspace.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        fft_forward.process(&mut nonlinear_product);
        let mut psi_prime: Vec<Complex64> = nonlinear_product
            .iter()
            .zip(roi.iter())
            .zip(meas_sqrt.iter().skip(j).step_by(dim))
            .map(|((&psi_val, &is_in_roi), &sg_val)| {
                if is_in_roi {
                    Complex64::from_polar(sg_val, psi_val.arg())
                } else {
                    ptychographic_threshold(psi_val, threshhold_gamma)
                }
            })
            .collect();
        fft_backward.process(&mut psi_prime);

        let new_component: Vec<Complex64> = psi_prime
            .iter()
            .zip(workspace.iter())
            .zip(field.iter())
            .map(|((&psi_val, &r), &field_val)| {
                r.conj() * (psi_val - r * field_val) / field_max_squared
            })
            .collect();

        for (field_val, &new_component_val) in field.iter_mut().zip(new_component.iter()) {
            *field_val += alpha * new_component_val;
        }
    }
    return (field.clone(), field);
}
