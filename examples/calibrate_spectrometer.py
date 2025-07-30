# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.14.13"
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
    measurement = aw.data.load_mean_spectrum_from_scarab(_path).to_normalized()
    return (measurement,)


@app.cell
def _(aw, lamp_options, mo):
    mo.stop(lamp_options.value is None)
    reference = aw.data.load_mean_spectrum_from_scarab(
        aw.spectrum.get_calibration_path() / lamp_options.value
    ).to_normalized()
    return (reference,)


@app.cell
def _():
    return


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
    Next, set the wavelength region-of-interest for the fitting, and the initial parameters of the response model. It is a slanted supergaussian. Minimize the difference between the calibrated measurement and the reference to give the fitting routine a good initial guess.

    Once it's decent, click run fitting.

    The "noise level" parameter will affect how the weights are adjusted based on the residuals of this fitting. If the noise level is zero, the weights will be set such that the calibrated spectrum exactly matches the reference, even for values where the measurement signal was very low. If the noise level is increased, the correction to the weights will be reduced for low signal-to-noise wavelengths.
    """
    )
    return


@app.cell
def _(measurement, mo, np):
    roi_lowest = mo.ui.number(
        value=np.min(measurement.wavelength_nm()),
        label="Shortest fitted wavelength (nm)",
    )
    roi_highest = mo.ui.number(
        value=np.max(measurement.wavelength_nm()),
        label="Longest fitted wavelength (nm)",
    )
    mo.output.append(roi_lowest)
    mo.output.append(roi_highest)

    return roi_highest, roi_lowest


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
    fit_cb = mo.ui.checkbox(label="Fit continuously")
    mo.output.append(amplitude_lam0)
    mo.output.append(amplitude_offset)
    mo.output.append(amplitude_slope)
    mo.output.append(amplitude_width)
    mo.output.append(amplitude_order)
    mo.output.append(wiener_noise_level)
    mo.output.append(fit_button)
    mo.output.append(fit_cb)
    return (
        amplitude_lam0,
        amplitude_offset,
        amplitude_order,
        amplitude_slope,
        amplitude_width,
        fit_button,
        fit_cb,
        wiener_noise_level,
    )


@app.cell
def _(
    amplitude_lam0,
    amplitude_offset,
    amplitude_order,
    amplitude_slope,
    amplitude_width,
    aw,
    roi_highest,
    roi_lowest,
    wavelength0,
    wavelength_offset,
    wavelength_slope,
    wiener_noise_level,
):
    input_parameters = aw.data.CalibrationInput(
        wavelength_center=wavelength0.value,
        wavelength_offset=wavelength_offset.value,
        wavelength_slope=wavelength_slope.value,
        amplitude_center=amplitude_lam0.value,
        amplitude_multiplier=amplitude_offset.value,
        amplitude_slope=amplitude_slope.value,
        amplitude_width=amplitude_width.value,
        amplitude_order=amplitude_order.value,
        noise_level=wiener_noise_level.value,
        roi_lowest=1e-9 * roi_lowest.value,
        roi_highest=1e-9 * roi_highest.value,
    )
    return (input_parameters,)


@app.cell
def _(
    aw,
    fit_button,
    fit_cb,
    input_parameters,
    measurement,
    plot_xmax,
    plot_xmin,
    reference,
):
    if fit_button.value or fit_cb.value:
        calibration_dataset = aw.data.CalibrationDataset.generate(
            measurement=measurement,
            reference=reference,
            input_parameters=input_parameters,
        )
        calibration_dataset.plot(plot_xmax=plot_xmax.value, plot_xmin=plot_xmin.value)
    else:
        input_parameters.plot(
            measurement=measurement,
            reference=reference,
            plot_xmax=plot_xmax.value,
            plot_xmin=plot_xmin.value,
        )
    aw.plot.showmo()
    return (calibration_dataset,)


@app.cell
def _(measurement, mo, np):
    plot_xmin = mo.ui.number(
        value=np.min(measurement.wavelength_nm()), label="Plot min wavelength (nm)"
    )
    plot_xmax = mo.ui.number(
        value=np.max(measurement.wavelength_nm()), label="Plot max wavelength (nm)"
    )
    mo.output.append(plot_xmin)
    mo.output.append(plot_xmax)
    return plot_xmax, plot_xmin


@app.cell
def _(mo):
    mo.md(r"""If the result is good, save it as an .npz file, and contribute it to the database :)""")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save calibration")
    save_dataset_button = mo.ui.run_button(label="Save dataset")
    mo.output.append(save_button)
    mo.output.append(save_dataset_button)
    return save_button, save_dataset_button


@app.cell
def _(calibration_dataset, filedialog, mo, save_button):
    mo.stop(not save_button.value)

    _file_path = filedialog.asksaveasfilename(
        title="Save File", filetypes=[("npz files", "*.npz")]
    )

    if _file_path is not None:
        try:
            calibration_dataset.final_calibration.save_npz(_file_path)
        except NameError:
            print("Can't save without data")

    return


@app.cell
def _(calibration_dataset, filedialog, mo, save_dataset_button):
    mo.stop(not save_dataset_button.value)

    _file_path = filedialog.asksaveasfilename(
        title="Save File", filetypes=[("YAML files", "*.yml")]
    )

    if _file_path is not None:
        try:
            calibration_dataset.save_yaml(_file_path)
        except NameError:
            print("Can't save without data")
    return


@app.cell
def _():
    import attoworld as aw
    import marimo as mo

    aw.plot.set_style("nick_dark")
    import tkinter as tk
    from tkinter import filedialog

    import numpy as np

    root = tk.Tk()
    root.withdraw()
    return aw, filedialog, mo, np


if __name__ == "__main__":
    app.run()
