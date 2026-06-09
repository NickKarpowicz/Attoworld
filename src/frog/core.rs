use super::apply_ptychographic_frog_iteration;
use pyo3::prelude::*;
use rand::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex};
use std::{f64, thread};

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum FrogType {
    Shg,
    PtychographicShg,
    Thg,
    Kerr,
    Xfrog,
    Blindfrog,
}

/// Core loop called by the wrapper above; see the comment there for a description of the inputs and outputs.
pub fn reconstruct_frog(
    measurement_sg_sqrt: &[f64],
    guess: Option<&[Complex64]>,
    trial_pulses: usize,
    iterations: usize,
    finishing_iterations: usize,
    frog_type: FrogType,
    spectrum: Option<&[f64]>,
    measured_gate: Option<&[Complex64]>,
    roi: Option<&[bool]>,
    ptycho_threshhold: Option<f64>,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let mut alloc = FrogAllocation::new(
        measurement_sg_sqrt,
        guess.map(|x| x.to_vec()),
        measured_gate.map(|x| x.to_vec()),
        frog_type.clone(),
        spectrum.map(|x| x.to_vec()),
        measured_gate.map(|x| x.to_vec()),
        roi.map(|x| x.to_vec()),
        ptycho_threshhold,
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
    if finishing_iterations == 0 {
        return (result.pulse.clone(), result.gate.clone(), result.error);
    }
    alloc.guess = Some(result.pulse.clone());
    alloc.gate_guess = Some(result.gate.clone());
    let last = reconstruct_frog_core(alloc, finishing_iterations);
    (last.pulse, last.gate, last.error)
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

#[derive(Clone)]
struct FrogAllocation {
    measurement_sg_sqrt: Vec<f64>,
    measurement_normalized: Vec<f64>,
    guess: Option<Vec<Complex64>>,
    gate_guess: Option<Vec<Complex64>>,
    frog_type: FrogType,
    spectrum: Option<Vec<f64>>,
    measured_gate: Option<Vec<Complex64>>,
    frequency_marginal: Vec<f64>,
    fft_forward: Arc<dyn Fft<f64>>,
    fft_backward: Arc<dyn Fft<f64>>,
    workspace: Vec<Complex64>,
    reconstructed_spectrogram: Vec<f64>,
    roi: Vec<bool>,
    ptycho_threshhold: f64,
}
impl FrogAllocation {
    fn new(
        measurement_sg_sqrt: &[f64],
        guess: Option<Vec<Complex64>>,
        gate_guess: Option<Vec<Complex64>>,
        frog_type: FrogType,
        spectrum: Option<Vec<f64>>,
        measured_gate: Option<Vec<Complex64>>,
        roi: Option<Vec<bool>>,
        ptycho_threshhold: Option<f64>,
    ) -> Self {
        let dim: usize = ((measurement_sg_sqrt.len() as f64).sqrt()).round() as usize;
        let measurement_sg_sqrt = measurement_sg_sqrt.to_vec();
        let workspace = vec![Complex64::new(0.0, 0.0); dim];
        let reconstructed_spectrogram = vec![0.0f64; dim * dim];
        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(dim);
        let fft_backward = planner.plan_fft_inverse(dim);
        let measurement_normalized = get_norm_meas(&measurement_sg_sqrt);
        let frequency_marginal = get_frequency_marginal(dim, &measurement_normalized);
        let roi = match roi {
            Some(r) => r,
            None => vec![true; dim],
        };
        let ptycho_threshhold = match ptycho_threshhold {
            Some(t) => t,
            None => 0.0,
        };

        FrogAllocation {
            measurement_sg_sqrt,
            measurement_normalized,
            guess,
            gate_guess,
            frog_type,
            spectrum,
            measured_gate,
            frequency_marginal,
            fft_forward,
            fft_backward,
            workspace,
            reconstructed_spectrogram,
            roi,
            ptycho_threshhold,
        }
    }
}

fn reconstruct_frog_core(mut alloc: FrogAllocation, iterations: usize) -> FrogResult {
    let mut pulse = match alloc.guess {
        Some(field) => field,
        None => generate_random_pulse_with_amplitude_spectrum(
            &alloc.frequency_marginal,
            alloc.fft_backward.clone(),
        ),
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
        (pulse, gate) = match alloc.frog_type {
            FrogType::PtychographicShg => apply_ptychographic_frog_iteration(
                &pulse,
                alloc.measurement_sg_sqrt.as_slice(),
                &mut alloc.workspace,
                alloc.fft_forward.clone(),
                alloc.fft_backward.clone(),
                &alloc.roi,
                alloc.ptycho_threshhold,
            ),
            _ => apply_frog_iteration(
                &pulse,
                &gate,
                &mut alloc.workspace,
                alloc.measurement_sg_sqrt.as_slice(),
                alloc.fft_forward.clone(),
                alloc.fft_backward.clone(),
            ),
        };
        pulse = frog_guess_from_pulse_and_gate(&pulse, &gate, &alloc.frog_type);
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
            workspace[i] = Complex64::from_polar(meas_sqrt[i * dim + j], workspace[i].arg());
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

/// Generate a new guess pulse based on the return pulse and gate from a simulation.
/// This is really only relevant for SHG frog, where they contain (in principle) identical
/// information, so the reconstruction can be improved by averaging them
fn frog_guess_from_pulse_and_gate(
    pulse: &[Complex64],
    gate: &[Complex64],
    nonlinearity: &FrogType,
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

/// Calculate the error of a FROG reconstruction
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

/// Resolve the relevant nonlinear process to provide the gate function associated with
/// a given pulse, or use the provided pulse to make a gate (xfrog)
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
        FrogType::PtychographicShg => {
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

/// Force the reconstruction to have a given intensity spectrum
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

/// get the norm of the measurement matrix
fn get_norm_meas(meas_sqrt: &[f64]) -> Vec<f64> {
    let norm: f64 = meas_sqrt.iter().map(|&a| a.powi(4)).sum::<f64>().sqrt();
    meas_sqrt.iter().map(|&a| a * a / norm).collect()
}

/// Generate a pulse whose spectrum is random (both intensity and phase), but with a
/// spectral amplitude probability given by the spectrum slice. In the current reconstruction
/// algorithm this spectrum is calculated from the frequency marginal of the measured spectrogram.
fn generate_random_pulse_with_amplitude_spectrum(
    spectrum: &[f64],
    fft_backward: Arc<dyn Fft<f64>>,
) -> Vec<Complex64> {
    let range = rand::distr::Uniform::new(0.0f64, f64::consts::TAU).unwrap();
    let mut rng = rand::rng();
    let mut complex_spectrum: Vec<Complex64> = spectrum
        .iter()
        .map(|&s| Complex64::from_polar(s * range.sample(&mut rng), range.sample(&mut rng)))
        .collect();
    fft_backward.process(&mut complex_spectrum);
    complex_spectrum
}

/// Get the frequency marginal of a spectrogram
fn get_frequency_marginal(dim: usize, spectrogram: &[f64]) -> Vec<f64> {
    spectrogram
        .chunks(dim)
        .map(|row| row.iter().sum::<f64>())
        .collect()
}
