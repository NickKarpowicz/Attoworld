"""Handle data sets related to producing a calibration, and complete the SpectrometerCalibration class."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy import constants

from ..numeric import interpolate
from .decorators import add_method, yaml_io
from .frog_data import Spectrogram
from .interop import IntensitySpectrum
from .spectrometer_calibration import SpectrometerCalibration


@add_method(SpectrometerCalibration, "apply_to_spectrum")
def calibration_apply_to_spectrum(self, spectrum_in):
    """Apply the calibration to an intensity spectrum.

    Args:
        self: the calibration (implicit)
        spectrum_in (IntensitySpectrum): the spectrum to be calibrated

    Returns:
        IntensitySpectrum: the calibrated spectrum

    """
    intensity_out = spectrum_in.spectrum * self.intensity_factors
    return IntensitySpectrum(
        spectrum=intensity_out,
        phase=spectrum_in.phase,
        wavelength=self.corrected_wavelengths,
        freq=self.corrected_frequencies,
    )


@add_method(SpectrometerCalibration, "apply_to_spectrogram")
def calibration_apply_to_spectrogram(self, spectrogram_in):
    """Apply the calibration to an intensity spectrum.

    Args:
        self: the calibration (implicit)
        spectrogram_in (Spectrogram): the spectrogram to be calibrated

    Returns:
        Spectrogram: the calibrated spectrogram

    """
    original_freqs = constants.speed_of_light / self.original_wavelengths
    original_freq_projection = interpolate(
        spectrogram_in.freq,
        original_freqs,
        self.corrected_frequencies,
        inputs_are_sorted=False,
    )
    data_out = np.zeros(spectrogram_in.data.shape, dtype=float)
    intensity_factors = interpolate(
        spectrogram_in.freq,
        self.corrected_frequencies,
        self.intensity_factors,
        inputs_are_sorted=False,
    )
    for i in range(spectrogram_in.data.shape[1]):
        data_out[:, i] = intensity_factors * interpolate(
            spectrogram_in.freq,
            original_freq_projection,
            np.array(spectrogram_in.data[:, i]),
        )

    return Spectrogram(
        data=data_out, freq=spectrogram_in.freq, time=spectrogram_in.time
    )


@yaml_io
@dataclass
class CalibrationInput:
    """Input parameters for fitting a calibration curve from a data set."""

    wavelength_center: float
    wavelength_offset: float
    wavelength_slope: float
    amplitude_center: float
    amplitude_multiplier: float
    amplitude_slope: float
    amplitude_width: float
    amplitude_order: float
    roi_lowest: float
    roi_highest: float
    noise_level: float

    def get_wavelength_array(self):
        """Produce an array of the wavelength parameters for fitting routines."""
        return np.array(
            [self.wavelength_center, self.wavelength_offset, self.wavelength_slope]
        )

    def get_amplitude_array(self):
        """Produce an array of the amplitude parameters for the fitting routine."""
        return np.array(
            [
                self.amplitude_center,
                self.amplitude_multiplier,
                self.amplitude_slope,
                self.amplitude_width,
                self.amplitude_order,
            ]
        )

    def plot(
        self,
        measurement: IntensitySpectrum,
        reference: IntensitySpectrum,
        plot_xmin=None,
        plot_xmax=None,
    ):
        """Combare the reference and measurement."""
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].plot(
            measurement.wavelength_nm(), measurement.spectrum, label="Measurement"
        )
        ax[0].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[0].set_xlabel("Wavelength (nm)")
        ax[0].set_ylabel("Intensity (Arb. unit)")
        ax[0].legend()
        initial_guess_calibration = generate_calibration_from_coeffs(
            self.get_amplitude_array(),
            self.get_wavelength_array(),
            measurement.wavelength,
        )

        initial_guess_spectrum = initial_guess_calibration.apply_to_spectrum(
            measurement
        )
        ax[1].plot(
            initial_guess_spectrum.wavelength_nm(),
            initial_guess_spectrum.spectrum,
            label="Initial guess",
        )
        ax[1].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[1].set_ylabel("Intensity (Arb. unit)")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].legend()

        if plot_xmin is not None and plot_xmax is not None:
            for a in ax:
                a.set_xlim(plot_xmin, plot_xmax)

        return fig


def generate_response_curve(
    wavelength: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Turn the parameter efficients array and wavelengths into a response curve."""
    relative_wl = wavelength - coefficients[0]
    taylor_series = coefficients[1] + 0.5 * relative_wl * coefficients[2]
    gaussian = np.exp(
        -(np.abs(relative_wl) ** coefficients[4])
        / (2 * coefficients[3] ** coefficients[4])
    )
    return taylor_series * gaussian


def get_new_wavelength(wavelengths_micron, taylor_coefficients_micron):
    """Apply the wavelength adjustment coefficients to the input wavelengths."""
    distance = wavelengths_micron - taylor_coefficients_micron[0]
    taylor_shift = taylor_coefficients_micron[1] * np.ones(distance.shape, dtype=float)
    for i in range(2, len(taylor_coefficients_micron)):
        taylor_shift += distance ** (i - 1) * taylor_coefficients_micron[i]

    new_wavelengths = 1e-6 * (wavelengths_micron + taylor_shift)
    new_freq = constants.speed_of_light / new_wavelengths
    return new_wavelengths, new_freq


def fit_calibration_amplitude_model(
    measurement, reference, wavelength_coeffs, amplitude_guess, roi
):
    """Fit the model calibration."""
    initial_cal = generate_calibration_from_coeffs(
        amplitude_guess, wavelength_coeffs, measurement.wavelength
    )
    reference_shift = reference.to_interpolated_wavelength(
        initial_cal.corrected_wavelengths
    )

    def residual_amp(coeff):
        cal = generate_calibration_from_coeffs(
            coeff, wavelength_coeffs, measurement.wavelength
        )
        calibrated = cal.apply_to_spectrum(measurement)
        residuals = calibrated.spectrum - reference_shift.spectrum
        return residuals[
            (calibrated.wavelength > roi[0]) & (calibrated.wavelength < roi[1])
        ]

    amplitude_result = scipy.optimize.least_squares(
        residual_amp, amplitude_guess, ftol=1e-12, max_nfev=16384
    )

    return amplitude_result.x


def generate_calibration_from_coeffs(amplitude_coeffs, wavelength_coeffs, wavelengths):
    """Generate a model calibration."""
    new_wavelengths, new_freqs = get_new_wavelength(
        wavelengths * 1e6, wavelength_coeffs
    )
    intensity_factors = 1.0 / generate_response_curve(
        wavelength=new_wavelengths * 1e6, coefficients=amplitude_coeffs
    )
    return SpectrometerCalibration(
        intensity_factors=intensity_factors,
        original_wavelengths=wavelengths,
        corrected_wavelengths=new_wavelengths,
        corrected_frequencies=new_freqs,
    )


@yaml_io
@dataclass
class CalibrationDataset:
    """Collects all of the data related to a spectrometer calibration."""

    measurement: IntensitySpectrum
    reference: IntensitySpectrum
    input_parameters: CalibrationInput
    parameterized_calibration: SpectrometerCalibration
    measurement_with_parameterized_calibration: IntensitySpectrum
    residuals: np.ndarray
    final_calibration: SpectrometerCalibration

    def plot(self, plot_xmin=None, plot_xmax=None):
        """Plot the calibration."""
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        ax[0, 0].plot(
            self.measurement.wavelength_nm(),
            self.measurement.spectrum,
            label="Measurement",
        )
        ax[0, 0].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 0].set_xlabel("Wavelength (nm)")
        ax[0, 0].set_ylabel("Intensity (Arb. unit)")
        ax[0, 0].legend()
        initial_guess_calibration = generate_calibration_from_coeffs(
            self.input_parameters.get_amplitude_array(),
            self.input_parameters.get_wavelength_array(),
            self.measurement.wavelength,
        )
        initial_guess_spectrum = initial_guess_calibration.apply_to_spectrum(
            self.measurement
        )
        ax[0, 1].plot(
            initial_guess_spectrum.wavelength_nm(),
            initial_guess_spectrum.spectrum,
            label="Initial guess",
        )
        ax[0, 1].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 1].set_ylabel("Intensity (Arb. unit)")
        ax[0, 1].set_xlabel("Wavelength (nm)")
        ax[0, 1].legend()

        ax[0, 2].plot(
            self.measurement_with_parameterized_calibration.wavelength_nm(),
            self.measurement_with_parameterized_calibration.spectrum,
            label="Model calibrated",
        )
        ax[0, 2].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[0, 2].set_ylabel("Intensity (Arb. unit)")
        ax[0, 2].set_xlabel("Wavelength (nm)")
        ax[0, 2].legend()

        ax[1, 0].plot(
            self.measurement_with_parameterized_calibration.wavelength_nm(),
            self.residuals,
            label="Residual",
        )
        ax[1, 0].set_ylim(-0.1, 0.1)
        ax[1, 0].set_ylabel("Intensity (Arb. unit)")
        ax[1, 0].set_xlabel("Wavelength (nm)")
        ax[1, 0].legend()

        second_calibration = self.final_calibration.apply_to_spectrum(self.measurement)
        ax[1, 1].plot(
            second_calibration.wavelength_nm(),
            second_calibration.spectrum,
            label="Calibrated",
        )
        ax[1, 1].plot(
            self.reference.wavelength_nm(), self.reference.spectrum, label="Reference"
        )
        ax[1, 1].set_ylabel("Intensity (Arb. unit)")
        ax[1, 1].set_xlabel("Wavelength (nm)")
        ax[1, 1].legend()
        ax[1, 2].plot(
            self.final_calibration.corrected_wavelengths * 1e9,
            self.final_calibration.intensity_factors,
            label="Final weights",
        )
        ax[1, 2].set_ylabel("Final weights")
        ax[1, 2].set_xlabel("Wavelength (nm)")
        ax[1, 2].legend()

        if plot_xmin is not None and plot_xmax is not None:
            for a in ax:
                for b in a:
                    b.set_xlim(plot_xmin, plot_xmax)

        return fig

    @staticmethod
    def generate(
        measurement: IntensitySpectrum,
        reference: IntensitySpectrum,
        input_parameters: CalibrationInput,
    ):
        """Create a SpectrometerCalibration from a measured lamp spectrum and reference spectrom, and a set of parameters.

        Args:
            measurement (IntensitySpectrum): Lamp spectrum measured with the spectrometer in question
            reference (IntensitySpectrum): Known reference spectrum for the lamp
            input_parameters (CalibrationInput): Set of input parameters for the calibration fitting

        Returns:
            SpectrometerCalibration: The resulting calibration

        """
        new_guess = fit_calibration_amplitude_model(
            measurement=measurement,
            reference=reference,
            wavelength_coeffs=input_parameters.get_wavelength_array(),
            amplitude_guess=input_parameters.get_amplitude_array(),
            roi=np.array([input_parameters.roi_lowest, input_parameters.roi_highest]),
        )

        parameterized_calibration = generate_calibration_from_coeffs(
            new_guess, input_parameters.get_wavelength_array(), measurement.wavelength
        )
        measurement_with_parameterized_calibration = (
            parameterized_calibration.apply_to_spectrum(measurement)
        )

        ref_on_new_axis = reference.to_interpolated_wavelength(
            measurement_with_parameterized_calibration.wavelength
        )
        measurement_shifted = measurement.to_corrected_wavelength(
            measurement_with_parameterized_calibration.wavelength
        )
        residuals = (
            measurement_with_parameterized_calibration.spectrum
            - ref_on_new_axis.spectrum
        )

        new_weights = (
            parameterized_calibration.intensity_factors
            - measurement_shifted.spectrum
            * residuals
            / (measurement_shifted.spectrum**2 + input_parameters.noise_level)
        )

        final_calibration = SpectrometerCalibration(
            intensity_factors=new_weights,
            original_wavelengths=measurement.wavelength,
            corrected_frequencies=parameterized_calibration.corrected_frequencies,
            corrected_wavelengths=parameterized_calibration.corrected_wavelengths,
        )

        return CalibrationDataset(
            measurement=measurement,
            reference=reference,
            input_parameters=input_parameters,
            parameterized_calibration=parameterized_calibration,
            measurement_with_parameterized_calibration=measurement_with_parameterized_calibration,
            residuals=residuals,
            final_calibration=final_calibration,
        )
