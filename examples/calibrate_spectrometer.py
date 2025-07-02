import marimo

__generated_with = "0.14.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Spectrometer calibration tool

    First, select the calibration lamp as measured by the spectrometer you want to calibrate.
    """
    )
    return


@app.cell
def _(mo):
    measurement_browser = mo.ui.file_browser(multiple=False)
    measurement_browser
    return (measurement_browser,)


@app.cell
def _(mo):
    mo.md(r"""Next, select the calibration lamp reference file from the list:""")
    return


@app.cell
def _(aw, mo):
    lamp_options = mo.ui.dropdown(
        options=[e.value for e in aw.spectrum.CalibrationLampReferences]
    )
    lamp_options
    return (lamp_options,)


@app.cell
def _(aw, measurement_browser, mo):
    _path = measurement_browser.path()
    mo.stop(_path is None)
    direct = aw.data.load_mean_spectrum_from_scarab(_path).to_normalized()
    return (direct,)


@app.cell
def _(aw, lamp_options, mo):
    mo.stop(lamp_options.value is None)
    reference = aw.data.load_mean_spectrum_from_scarab(
        aw.spectrum.get_calibration_path() / lamp_options.value
    ).to_normalized()
    return (reference,)


@app.cell
def _(aw, np, scipy):
    def generate_response_curve(
        wavelength: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        relative_wl = wavelength - coefficients[0]
        taylor_series = coefficients[1] + 0.5 * relative_wl * coefficients[2]
        gaussian = np.exp(
            -(np.abs(relative_wl) ** coefficients[4])
            / (2 * coefficients[3] ** coefficients[4])
        )
        return taylor_series * gaussian

    def get_new_wavelength(wavelengths_micron, taylor_coefficients_micron):
        distance = wavelengths_micron - taylor_coefficients_micron[0]
        taylor_shift = taylor_coefficients_micron[1] * np.ones(
            distance.shape, dtype=float
        )
        for i in range(2, len(taylor_coefficients_micron)):
            taylor_shift += distance ** (i - 1) * taylor_coefficients_micron[i]

        new_wavelengths = 1e-6 * (wavelengths_micron + taylor_shift)
        new_freq = scipy.constants.speed_of_light / new_wavelengths
        return new_wavelengths, new_freq

    def fit_calibration_amplitude_model(
        measurement, reference, wavelength_coeffs, amplitude_guess, roi
    ):
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

    def generate_calibration_from_coeffs(
        amplitude_coeffs, wavelength_coeffs, wavelengths
    ):
        new_wavelengths, new_freqs = get_new_wavelength(
            wavelengths * 1e6, wavelength_coeffs
        )
        intensity_factors = 1.0 / generate_response_curve(
            wavelength=new_wavelengths * 1e6, coefficients=amplitude_coeffs
        )
        return aw.data.SpectrometerCalibration(
            intensity_factors=intensity_factors,
            corrected_wavelengths=new_wavelengths,
            corrected_frequencies=new_freqs,
        )

    return fit_calibration_amplitude_model, generate_calibration_from_coeffs


@app.cell
def _(mo):
    mo.md(r"""Now, manually adjust the wavelength axis (optional - if it's not necessary, set offset and slope to zero)""")
    return


@app.cell
def _(mo):
    wavelength0 = mo.ui.slider(
        value=0.75,
        start=0,
        stop=3,
        step=1e-4,
        label="Wavelength shift center",
        show_value=True,
    )
    wavelength_offset = mo.ui.slider(
        value=-0.0107,
        start=-1.0,
        stop=1.0,
        step=1e-4,
        label="Wavelength offset",
        show_value=True,
    )
    wavelength_slope = mo.ui.slider(
        value=2e-3,
        start=-1,
        stop=1,
        step=1e-3,
        label="Wavelength slope",
        show_value=True,
    )
    mo.output.append(wavelength0)
    mo.output.append(wavelength_offset)
    mo.output.append(wavelength_slope)
    return wavelength0, wavelength_offset, wavelength_slope


@app.cell
def _(mo):
    mo.md(
        r"""
    Next, set the parameters of the response model. It is a slanted supergaussian. Minimize the difference between the calibrated measurement and the reference to give the fitting routine a good initial guess.

    Once it's decent, click run fitting.

    The "noise level" parameter will affect how the weights are adjusted based on the residuals of this fitting. If the noise level is zero, the weights will be set such that the calibrated spectrum exactly matches the reference, even for values where the measurement signal was very low. If the noise level is increased, the correction to the weights will be reduced for low signal-to-noise wavelengths.
    """
    )
    return


@app.cell
def _(mo):
    amplitude_lam0 = mo.ui.slider(
        value=0.65,
        start=0,
        stop=3,
        step=1e-4,
        label="Amplitude correction center",
        show_value=True,
    )
    amplitude_offset = mo.ui.slider(
        value=3.0,
        start=0,
        stop=30,
        step=1e-4,
        label="Amplitude multiplier",
        show_value=True,
    )
    amplitude_slope = mo.ui.slider(
        value=4.0,
        start=-30,
        stop=30,
        step=1e-4,
        label="Amplitude slope",
        show_value=True,
    )
    amplitude_width = mo.ui.slider(
        value=0.32, start=0, stop=3, step=1e-4, label="Amplitude width", show_value=True
    )
    amplitude_order = mo.ui.slider(
        value=4.0,
        start=2,
        stop=32,
        step=1e-4,
        label="Amplitude Gaussian order",
        show_value=True,
    )
    wiener_noise_level = mo.ui.slider(
        value=0.01, start=0.0, stop=0.1, step=1e-4, label="Noise level", show_value=True
    )
    fit_button = mo.ui.run_button(label="Run fitting")
    mo.output.append(amplitude_lam0)
    mo.output.append(amplitude_offset)
    mo.output.append(amplitude_slope)
    mo.output.append(amplitude_width)
    mo.output.append(amplitude_order)
    mo.output.append(wiener_noise_level)
    mo.output.append(fit_button)

    return (
        amplitude_lam0,
        amplitude_offset,
        amplitude_order,
        amplitude_slope,
        amplitude_width,
        fit_button,
        wiener_noise_level,
    )


@app.cell
def _(
    amplitude_lam0,
    amplitude_offset,
    amplitude_order,
    amplitude_slope,
    amplitude_width,
    direct,
    generate_calibration_from_coeffs,
    np,
    wavelength0,
    wavelength_offset,
    wavelength_slope,
):
    initial_guess = np.array(
        [
            amplitude_lam0.value,
            amplitude_offset.value,
            amplitude_slope.value,
            amplitude_width.value,
            amplitude_order.value,
        ]
    )
    wavelength_guess = [
        wavelength0.value,
        wavelength_offset.value,
        wavelength_slope.value,
    ]
    guess_cal = generate_calibration_from_coeffs(
        initial_guess, wavelength_guess, direct.wavelength
    )
    first_guess = guess_cal.apply_to_spectrum(direct)

    return first_guess, initial_guess, wavelength_guess


@app.cell
def _(
    aw,
    direct,
    first_guess,
    fit_button,
    fit_calibration_amplitude_model,
    generate_calibration_from_coeffs,
    initial_guess,
    np,
    plt,
    reference,
    wavelength_guess,
    wiener_noise_level,
):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    print(ax.shape)
    ax[0, 0].plot(direct.wavelength_nm(), direct.spectrum, label="Measurement")
    ax[0, 0].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
    ax[0, 0].set_xlabel("Wavelength (nm)")
    ax[0, 0].set_ylabel("Intensity (Arb. unit)")
    ax[0, 0].legend()
    ax[0, 1].plot(
        first_guess.wavelength_nm(), first_guess.spectrum, label="Initial guess"
    )
    ax[0, 1].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
    ax[0, 1].set_ylabel("Intensity (Arb. unit)")
    ax[0, 1].set_xlabel("Wavelength (nm)")
    ax[0, 1].legend()
    if fit_button.value:
        new_guess = fit_calibration_amplitude_model(
            measurement=direct,
            reference=reference,
            wavelength_coeffs=wavelength_guess,
            amplitude_guess=initial_guess,
            roi=np.array([200e-9, 950e-9]),
        )
        parametrized_cal = generate_calibration_from_coeffs(
            new_guess, wavelength_guess, direct.wavelength
        )
        calibrated = parametrized_cal.apply_to_spectrum(direct)

        ax[0, 2].plot(
            calibrated.wavelength_nm(), calibrated.spectrum, label="Model calibrated"
        )
        ax[0, 2].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[0, 2].set_ylabel("Intensity (Arb. unit)")
        ax[0, 2].set_xlabel("Wavelength (nm)")
        ax[0, 2].legend()
        ref_on_new_axis = reference.to_interpolated_wavelength(calibrated.wavelength)
        A = ref_on_new_axis.spectrum
        B = calibrated.spectrum
        ax[1, 0].plot(calibrated.wavelength_nm(), A - B, label="Residual")
        ax[1, 0].set_ylim(-0.1, 0.1)
        ax[1, 0].set_ylabel("Intensity (Arb. unit)")
        ax[1, 0].set_xlabel("Wavelength (nm)")
        ax[1, 0].legend()
        direct_shifted = direct.to_corrected_wavelength(calibrated.wavelength)
        residual = B - A
        new_weights = (
            direct_shifted.spectrum
            * (calibrated.spectrum - residual)
            / (direct_shifted.spectrum**2 + wiener_noise_level.value)
        )
        new_weights = (
            parametrized_cal.intensity_factors
            - direct_shifted.spectrum
            * residual
            / (direct_shifted.spectrum**2 + wiener_noise_level.value)
        )
        adjusted_cal = aw.data.SpectrometerCalibration(
            intensity_factors=new_weights,
            corrected_frequencies=parametrized_cal.corrected_frequencies,
            corrected_wavelengths=parametrized_cal.corrected_wavelengths,
        )
        second_calibration = adjusted_cal.apply_to_spectrum(direct)
        ax[1, 1].plot(
            second_calibration.wavelength_nm(),
            second_calibration.spectrum,
            label="Calibrated",
        )
        ax[1, 1].plot(reference.wavelength_nm(), reference.spectrum, label="Reference")
        ax[1, 1].set_ylabel("Intensity (Arb. unit)")
        ax[1, 1].set_xlabel("Wavelength (nm)")
        ax[1, 1].legend()
        ax[1, 2].plot(
            adjusted_cal.corrected_wavelengths * 1e9,
            adjusted_cal.intensity_factors,
            label="Final weights",
        )
        ax[1, 2].set_ylabel("Final weights")
        ax[1, 2].set_xlabel("Wavelength (nm)")
        ax[1, 2].legend()
    aw.plot.showmo()
    return (adjusted_cal,)


@app.cell
def _(mo):
    mo.md(r"""If the result is good, save it as an .npz file, and contribute it to the database :)""")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save calibration")
    save_button
    return (save_button,)


@app.cell
def _(adjusted_cal, filedialog, mo, save_button):
    mo.stop(not save_button.value)

    _file_path = filedialog.asksaveasfilename(
        title="Save File", filetypes=[("npz files", "*.npz")]
    )

    if _file_path is not None:
        try:
            adjusted_cal.save_npz(_file_path)
        except NameError:
            print("Can't save without data")

    return


@app.cell
def _():
    import marimo as mo
    import attoworld as aw
    aw.plot.set_style('nick_dark')
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    return aw, filedialog, mo, np, plt, scipy


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
