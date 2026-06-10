# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "attoworld==2026.2.5",
#     "marimo>=0.23.9",
#     "numpy==2.4.6",
# ]
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Spectrometer calibration tool

    First, select the calibration lamp as measured by the spectrometer you want to calibrate.
    """)
    return


@app.cell
def _(mo):
    measurement_file = mo.ui.file(label="Measurement file")
    mo.output.append(measurement_file)
    return (measurement_file,)


@app.cell
def _(mo):
    measurement_format = mo.ui.dropdown(options=["2 Columns (nm | Intensity)", "Text with headers"], value="2 Columns (nm | Intensity)")
    mo.output.append(measurement_format)
    return (measurement_format,)


@app.cell
def _(measurement_file):
    measurement_data = measurement_file.contents()
    return (measurement_data,)


@app.cell
def _(measurement_data, measurement_format, mo):
    measurement_wavelength_header = mo.ui.text(label="Wavelength column key:", value="wavelength (nm)")
    measurement_wavelength_multiplier = mo.ui.number(label="Wavelength multiplier:", value=1e9)
    measurement_intensity_header = mo.ui.text(label="Intensity column key:", value="intensity (a.u.)")
    measurement_skip_lines = mo.ui.number(value=0, label="Header lines:")
    if measurement_data is not None:
        if measurement_format.value == "Text with headers":
            mo.output.append(measurement_wavelength_header)
            mo.output.append(measurement_wavelength_multiplier)
            mo.output.append(measurement_intensity_header)
        if measurement_format.value == "2 Columns (nm | Intensity)":
            mo.output.append(measurement_skip_lines)
    return (
        measurement_intensity_header,
        measurement_skip_lines,
        measurement_wavelength_header,
        measurement_wavelength_multiplier,
    )


@app.cell
def _(
    aw,
    measurement_data,
    measurement_format,
    measurement_intensity_header,
    measurement_skip_lines,
    measurement_wavelength_header,
    measurement_wavelength_multiplier,
):
    measurement = None
    if measurement_data is not None:
        match measurement_format.value:
            case "2 Columns (nm | Intensity)":
                measurement = aw.data.load_mean_spectrum_from_scarab(
                    measurement_data.decode("utf-8"), is_data_string=True, header_size=measurement_skip_lines.value
                )
            case "Text with headers":
                measurement = aw.data.load_spectrum_from_text(
                    filename_or_data_string=measurement_data.decode("utf-8"),
                    wavelength_multiplier=1.0
                    / measurement_wavelength_multiplier.value,
                    wavelength_field=measurement_wavelength_header.value,
                    spectrum_field=measurement_intensity_header.value,
                    is_data_string=True
                )
        measurement.spectrum[measurement.spectrum<0] = 0.0
    return (measurement,)


@app.cell
def _(aw, measurement, mo):
    if measurement is not None:
        mo.output.append(mo.md("### Measurement:"))
        mo.output.append(mo.md("Check that the measurement loaded correctly."))
        measurement.plot_with_group_delay()
        aw.plot.showmo()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Next, select the calibration lamp reference file from the list:
    """)
    return


@app.cell
def _(aw, mo):
    lamp_options = mo.ui.dropdown(
        options=[e.value for e in aw.spectrum.CalibrationLampReferences]
    )
    lamp_options
    return (lamp_options,)


@app.cell
def _(aw, lamp_options, mo):
    mo.stop(lamp_options.value is None)
    reference = aw.data.load_mean_spectrum_from_scarab(
        aw.spectrum.get_calibration_path() / lamp_options.value
    ).to_normalized()
    return (reference,)


@app.cell
def _(mo):
    mo.md(r"""
    Now, manually adjust the wavelength axis (optional - if it's not necessary, set offset and slope to zero

    Next, set the wavelength region-of-interest for the fitting, and the initial parameters of the response model. It is a slanted supergaussian. Minimize the difference between the calibrated measurement and the reference to give the fitting routine a good initial guess.

    Once it's decent, click run fitting.

    The "noise level" parameter will affect how the weights are adjusted based on the residuals of this fitting. If the noise level is zero, the weights will be set such that the calibrated spectrum exactly matches the reference, even for values where the measurement signal was very low. If the noise level is increased, the correction to the weights will be reduced for low signal-to-noise wavelengths.
    """)
    return


@app.cell
def _(mo):
    wavelength0 = mo.ui.number(
        value=0.75,
        start=0,
        stop=3,
        step=1e-4,
        label="Wavelength shift center",
    )
    wavelength_offset = mo.ui.number(
        value=-0.0107,
        start=-1.0,
        stop=1.0,
        step=1e-4,
        label="Wavelength offset",
    )
    wavelength_slope = mo.ui.number(
        value=2e-3,
        start=-1,
        stop=1,
        step=1e-3,
        label="Wavelength slope",
    )
    mo.output.append(wavelength0)
    mo.output.append(wavelength_offset)
    mo.output.append(wavelength_slope)
    return wavelength0, wavelength_offset, wavelength_slope


@app.cell
def _(measurement, mo, np):
    if measurement is not None:
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
    amplitude_lam0 = mo.ui.number(
        value=0.65,
        start=0,
        stop=3,
        step=1e-4,
        label="Amplitude correction center",
    )
    amplitude_offset = mo.ui.number(
        value=3.0,
        start=0,
        stop=30,
        step=1e-4,
        label="Amplitude multiplier",
    )
    amplitude_slope = mo.ui.number(
        value=4.0,
        start=-30,
        stop=30,
        step=1e-4,
        label="Amplitude slope",
    )
    amplitude_width = mo.ui.number(
        value=0.32, start=0, stop=3, step=1e-4, label="Amplitude width"
    )
    amplitude_order = mo.ui.number(
        value=4.0,
        start=2,
        stop=32,
        step=1e-4,
        label="Amplitude Gaussian order",
    )
    wiener_noise_level = mo.ui.number(
        value=0.01, start=0.0, stop=0.1, step=1e-4, label="Noise level"
    )

    mo.output.append(amplitude_lam0)
    mo.output.append(amplitude_offset)
    mo.output.append(amplitude_slope)
    mo.output.append(amplitude_width)
    mo.output.append(amplitude_order)
    mo.output.append(wiener_noise_level)
    return (
        amplitude_lam0,
        amplitude_offset,
        amplitude_order,
        amplitude_slope,
        amplitude_width,
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
    measurement,
    roi_highest,
    roi_lowest,
    wavelength0,
    wavelength_offset,
    wavelength_slope,
    wiener_noise_level,
):
    if measurement is not None:
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
def _(aw, input_parameters, measurement, plot_xmax, plot_xmin, reference):
    if measurement is not None:
        calibration_dataset = aw.data.CalibrationDataset.generate(
            measurement=measurement,
            reference=reference,
            input_parameters=input_parameters,
        )
        calibration_dataset.plot(plot_xmax=plot_xmax.value, plot_xmin=plot_xmin.value)
        aw.plot.showmo()
    return (calibration_dataset,)


@app.cell
def _(measurement, mo, np):
    if measurement is not None:
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
    mo.md(r"""
    If the result is good, save it as an .npz file, and contribute it to the database :)
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save calibration .npz file")
    save_dataset_button = mo.ui.run_button(label="Save dataset archive")
    mo.output.append(save_button)
    mo.output.append(save_dataset_button)
    return save_button, save_dataset_button


@app.cell
def _(calibration_dataset, display_download_link_from_file, mo, save_button):
    mo.stop(not save_button.value)
    calibration_dataset.final_calibration.save_npz("calibration_result.npz")
    display_download_link_from_file(
        path="calibration_result.npz",
        output_name="calibration_result.npz",
        mime_type="application/zip",
    )
    # _file_path = filedialog.asksaveasfilename(
    #     title="Save File", filetypes=[("npz files", "*.npz")]
    # )

    # if _file_path is not None:
    #     try:
    #         calibration_dataset.final_calibration.save_npz(_file_path)
    #     except NameError:
    #         print("Can't save without data")
    return


@app.cell
def _(
    calibration_dataset,
    display_download_link_from_file,
    mo,
    save_dataset_button,
):
    mo.stop(not save_dataset_button.value)

    calibration_dataset.save_yaml("calibration_dataset.yml")
    display_download_link_from_file(
        path="calibration_dataset.yml",
        output_name="calibration_dataset.yml",
        mime_type="text/yaml",
    )
    # _file_path = filedialog.asksaveasfilename(
    #     title="Save File", filetypes=[("YAML files", "*.yml")]
    # )

    # if _file_path is not None:
    #     try:
    #         calibration_dataset.save_yaml(_file_path)
    #     except NameError:
    #         print("Can't save without data")
    return


@app.cell
def _():
    import attoworld as aw
    import marimo as mo

    aw.plot.set_style("nick_dark")


    import numpy as np

    def display_download_link_from_file(
        path, output_name, mime_type="text/plain"
    ):
        with open(path, "rb") as _file:
            mo.output.append(
                mo.download(
                    data=_file,
                    filename=output_name,
                    mimetype=mime_type,
                    label=f"Download {output_name}",
                )
            )

    return aw, display_download_link_from_file, mo, np


if __name__ == "__main__":
    app.run()
