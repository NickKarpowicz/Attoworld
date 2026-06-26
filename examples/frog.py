# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "attoworld>=2026.2.8",
#     "marimo>=0.23.10",
#     "numpy>=2.4.6",
#     "pyside6; sys_platform != 'emscripten'",
# ]
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="Frog")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import attoworld as aw
    import numpy as np
    import pathlib
    import time

    return aw, mo, np, pathlib, plt, time


@app.cell
def _(mo):
    # check if running in a browser for extra setup
    import sys
    is_in_web_notebook = sys.platform == "emscripten"
    if is_in_web_notebook:
        import zipfile
        mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10000000000
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
    else:
        from PySide6.QtWidgets import QApplication, QFileDialog
        qtapp = QApplication(sys.argv)
    return (
        QFileDialog,
        display_download_link_from_file,
        is_in_web_notebook,
        zipfile,
    )


@app.cell
def _(mo):
    help_cb = mo.ui.checkbox(value=True, label="show instructions")
    plot_style_selector = mo.ui.dropdown(options=["Dark", "Light"], label="Plot style", value="Dark")
    mo.output.append(plot_style_selector)
    mo.output.append(help_cb)
    return help_cb, plot_style_selector


@app.cell
def _(help_cb, mo):
    mo.output.append(mo.md("# FROG Reconstruction:"))
    if help_cb.value:
        mo.output.append(mo.md("This code will take a measured FROG (field-resolved optical gating) trace and attempt to reconstruct it. The reconstruction is written in Rust with a python interface, and can be found at the [Attoworld github repo](https://github.com/NickKarpowicz/Attoworld)."))
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Select your FROG file:"))
    if help_cb.value:
        mo.output.append(mo.md("Currently only the .dwc file format is set up, but I can easily add different import methods if you send me an example of the file you want to reconstruct!"))
    return


@app.cell
def _(mo):
    file_browser = mo.ui.file(filetypes=[".dwc"], label="Select .dwc file")
    file_browser
    return (file_browser,)


@app.cell
def _(help_cb, mo):
    if help_cb.value:
        mo.output.append(mo.md("If your spectrometer is in our list (i.e. you're in the MPQ lab and calibrated your spectrometer), it will be in the list. If you want your reconstruction to work better, you can send a calibration result to me, using the calibration script in the Attoworld repo examples folder."))
    return


@app.cell
def _(aw, mo):
    calibration_selector = mo.ui.dropdown(options=[e.value for e in aw.spectrum.CalibrationData],label="Calibration:")
    mo.output.append(calibration_selector)
    return (calibration_selector,)


@app.cell
def _(mo):
    mode_selector = mo.ui.dropdown(options=["SHG GP", "SHG ptychographic", "THG", "Kerr", "XFROG", "BlindFROG"], label="FROG type:", value="SHG ptychographic")
    mode_selector
    return (mode_selector,)


@app.cell
def _(mo, mode_selector):
    xfrog_reference_file = mo.ui.file(filetypes=[".yml",".dat"], label="Select reference file")
    xfrog_time_reverse_checkbox = mo.ui.checkbox(label="Reverse time")
    if(mode_selector.value == "XFROG"):
        mo.output.append(mo.md("---"))
        mo.output.append(mo.md("### XFROG Reference:"))
        mo.output.append(xfrog_reference_file)
        mo.output.append(xfrog_time_reverse_checkbox)
    return xfrog_reference_file, xfrog_time_reverse_checkbox


@app.cell
def _(
    aw,
    mo,
    mode_selector,
    np,
    pathlib,
    xfrog_reference_file,
    xfrog_time_reverse_checkbox,
):
    if((mode_selector.value == "XFROG") and (xfrog_reference_file.name() is not None)):
        mo.output.append(mo.md("### Loaded reference:"))
        _type = pathlib.Path(xfrog_reference_file.name()).suffix
        if _type == ".yml":
            xfrog_reference = aw.data.FrogData.load_yaml_bytestream(xfrog_reference_file.contents())
            if xfrog_time_reverse_checkbox.value:
                xfrog_reference.raw_reconstruction = np.flipud(xfrog_reference.raw_reconstruction)
                xfrog_reference.pulse.wave = np.flipud(xfrog_reference.pulse.wave)
                xfrog_reference.spectrum.spectrum = np.conj(xfrog_reference.spectrum.spectrum)
            xfrog_reference.plot_all(figsize=(9,6))
            aw.plot.showmo()
    else:
        xfrog_reference=None
    return (xfrog_reference,)


@app.cell
def _(aw, calibration_selector, file_browser, mo, np):
    _path = file_browser.contents()
    if _path is not None:
        input_data = aw.data.read_dwc(file_or_path=_path, is_buffer=True)
        if calibration_selector.value is not None:
            calibration = aw.data.SpectrometerCalibration.from_npz(
                aw.spectrum.get_calibration_path() / calibration_selector.value
            )
            input_data = calibration.apply_to_spectrogram(input_data)
        mo.output.append(mo.md("Loaded spectrogram parameters:"))
        mo.output.append(mo.md(f"    time step: {np.mean(np.diff(input_data.time)) * 1e15:.1g} fs"))
        mo.output.append(mo.md(f"    frequency step: {np.mean(np.diff(input_data.freq)) * 1e-12:.1g} THz"))
        mo.output.append(mo.md(f"    central frequency: {np.mean(input_data.freq) * 1e-12:.1f} THz"))
    else:
        input_data = None
    return (input_data,)


@app.cell
def _(help_cb, mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Bin data onto evenly spaced grid:"))
    if help_cb.value:
        mo.output.append(mo.md("""The box should contain your full trace, with some empty space on all sides. Make the time-step smaller to have a larger frequency range. Increase the time range by increasing the box size.

    Increasing the dark noise level parameter will gate the spectrogram, reducing noise in the final trace, but be careful that you don't remove actual signal.

    The block-averaging options will bin pixels before interpolation onto the final grid, and can help to improve the signal-to-noise ratio as long as the settings don't reduce the resolution. Median blocking will take the median rather than average, for use when the data has larger outliers.

    The spatial chirp correction will un-tilt a tilted spectrogram, but is only for checking: if it helps, the measurement isn't likely to be valid."""))
    return


@app.cell
def _(mo):
    bin_loaded_file = mo.ui.file(filetypes=[".yml"], label="Load settings from .yml")
    bin_loaded_file
    return (bin_loaded_file,)


@app.cell
def _(aw, bin_loaded_file):
    _contents = bin_loaded_file.contents()
    if _contents is not None:
        loaded_settings = aw.data.FrogBinSettings.load_yaml_bytestream(_contents)
    else:
        loaded_settings = aw.data.FrogBinSettings(
            size=96,
            dt=3e-15,
            t0=0.0,
            auto_t0=True,
            f0=740e12,
            dc_offset=0.0002,
            freq_binning=1,
            time_binning=1,
            median_binning=False,
            spatial_chirp_correction=False,
        )
    return (loaded_settings,)


@app.cell
def _(is_in_web_notebook, loaded_settings, mo):
    bin_size = mo.ui.number(label="size", value=loaded_settings.size, step=2)
    bin_dt = mo.ui.number(label="dt (fs)", value=loaded_settings.dt*1e15, step=0.1)
    bin_t0 = mo.ui.number(label="t0 (fs)", value=loaded_settings.t0 * 1e-15, step=0.1)
    bin_t0_auto = mo.ui.checkbox(label="Auto time centering", value=loaded_settings.auto_t0)
    bin_f0 = mo.ui.number(label="f0 (THz)", value=loaded_settings.f0*1e-12, step=0.1)
    bin_offset = mo.ui.number(label="dark noise level", value=loaded_settings.dc_offset, step=1e-5)
    bin_fblock = mo.ui.number(label="freq block avg.", start=1, value=loaded_settings.freq_binning, step=1)
    bin_tblock = mo.ui.number(label="time block avg.", start=1, value=loaded_settings.time_binning, step=1)
    bin_median = mo.ui.checkbox(label="median blocking", value=loaded_settings.median_binning)
    bin_spatial_chirp_correction = mo.ui.checkbox(label="correct spatial chirp", value=loaded_settings.spatial_chirp_correction)
    bin_log = mo.ui.checkbox(value=True, label="log scale")
    bin_geometric_correction = mo.ui.checkbox(label="correct geometric smearing")
    bin_geometric_smearing_angle = mo.ui.number(value = 0.0, step=0.001, start=0, stop=180, label="beam angle (deg)")
    bin_geometric_smearing_waist = mo.ui.number(value=10.0, step=0.1, start=0.1, label="beam waist (microns)")
    bin_geometric_smearing_max = mo.ui.number(value=10, step=0.1, start=1, label="max amplifcation factor")
    mo.output.append(bin_size)
    mo.output.append(bin_dt)
    mo.output.append(bin_t0)
    mo.output.append(bin_t0_auto)
    mo.output.append(bin_f0)
    mo.output.append(bin_offset)
    mo.output.append(bin_fblock)
    mo.output.append(bin_tblock)
    mo.output.append(bin_median)
    mo.output.append(bin_spatial_chirp_correction)


    if not is_in_web_notebook:
        bin_save_button = mo.ui.run_button(label="Save settings")
        mo.output.append(bin_save_button)
    return (
        bin_dt,
        bin_f0,
        bin_fblock,
        bin_geometric_correction,
        bin_geometric_smearing_angle,
        bin_geometric_smearing_max,
        bin_geometric_smearing_waist,
        bin_log,
        bin_median,
        bin_offset,
        bin_save_button,
        bin_size,
        bin_spatial_chirp_correction,
        bin_t0,
        bin_t0_auto,
        bin_tblock,
    )


@app.cell
def _(
    aw,
    bin_dt,
    bin_f0,
    bin_fblock,
    bin_median,
    bin_offset,
    bin_size,
    bin_spatial_chirp_correction,
    bin_t0,
    bin_t0_auto,
    bin_tblock,
):
    if bin_t0_auto.value:
        _t0 = None
    else:
        _t0 = bin_t0.value * 1e-15
    bin_settings = aw.data.FrogBinSettings(
        size=int(bin_size.value),
        dt=bin_dt.value * 1e-15,
        t0=bin_t0.value * 1e-15,
        auto_t0=bin_t0_auto.value,
        f0=bin_f0.value * 1e12,
        dc_offset=bin_offset.value,
        time_binning=int(bin_tblock.value),
        freq_binning=int(bin_fblock.value),
        median_binning=bool(bin_median.value),
        spatial_chirp_correction=bool(bin_spatial_chirp_correction.value),
    )
    return (bin_settings,)


@app.cell
def _(
    bin_geometric_correction,
    bin_geometric_smearing_angle,
    bin_geometric_smearing_max,
    bin_geometric_smearing_waist,
    bin_settings,
    display_download_link_from_file,
    input_data,
    is_in_web_notebook,
):
    if input_data is not None:
        if bin_geometric_correction.value:
            frog_data = input_data.to_bin_pipeline_result(bin_settings).to_deconvolved_geometric_smearing(
                angle_in_degrees=bin_geometric_smearing_angle.value,
                beamwaist_meters=1e-6 * bin_geometric_smearing_waist.value,
                max_amplification_factor=bin_geometric_smearing_max.value
            )
        else:
            frog_data = input_data.to_bin_pipeline_result(bin_settings)

        if is_in_web_notebook:
            bin_settings.save_yaml("bin_settings.yml")
            display_download_link_from_file(
                path="bin_settings.yml",
                output_name="bin_settings.yml",
                mime_type="text/yaml",
            )
    else:
        frog_data = None
    return (frog_data,)


@app.cell
def _(aw, bin_log, frog_data, plot_style_selector):
    # if not bin_live.value:
    #     mo.stop(not bin_button.value)
    if frog_data is not None:
        if plot_style_selector.value == "Light":
            aw.plot.set_style("light")
        else:
            aw.plot.set_style("nick_dark")


        if bin_log.value:
            frog_data.plot_log()
        else:
            frog_data.plot()
        aw.plot.showmo()
    return


@app.cell
def _(bin_geometric_correction, bin_log, mo):
    mo.output.append(bin_log)
    mo.output.append(bin_geometric_correction)
    return


@app.cell
def _(
    bin_geometric_correction,
    bin_geometric_smearing_angle,
    bin_geometric_smearing_max,
    bin_geometric_smearing_waist,
    help_cb,
    mo,
):
    if bin_geometric_correction.value:
        mo.output.append(mo.md("### Geometric smearing correction parameters"))
        if help_cb.value:
            mo.output.append(mo.md("This experimental feature will correct for blurring along the delay-axis caused by the angle between the beams in the FROG setup. Since this essentially a deconvolution, it will increase high-frequency noise, which may be suppressed by lowering the max amplification value."))
        mo.output.append(bin_geometric_smearing_angle)
        mo.output.append(bin_geometric_smearing_waist)
        mo.output.append(bin_geometric_smearing_max)
    return


@app.cell
def _(
    QFileDialog,
    bin_save_button,
    bin_settings,
    is_in_web_notebook,
    mo,
    pathlib,
):

    if not is_in_web_notebook and bin_settings is not None:
        mo.stop(not bin_save_button.value)
        _file_path, _file_type = QFileDialog.getSaveFileName(
                None, "Save file", "", "YAML Files (*.yml)")
        if (_file_path is not None) and (bin_settings is not None) and (_file_path != ""):
            if pathlib.Path(_file_path).suffix is "":
                _file_path += ".yml"
            bin_settings.save_yaml(_file_path)
    return


@app.cell
def _(help_cb, mo, mode_selector):
    ptycho_roi_lower = mo.ui.number(value = 300.0, step=0.1, label="ROI lower frequency (THz)")
    ptycho_roi_upper = mo.ui.number(value = 900.0, step=0.1, label="ROI upper frequency (THz)")
    ptycho_exclude_lower = mo.ui.number(value = 1000.0, step=0.1, label="Excluded region lower frequency (THz)")
    ptycho_exclude_upper = mo.ui.number(value = 1200.0, step=0.1, label="Excluded region upper frequency (THz)")
    ptycho_threshhold = mo.ui.number(value = 10000.0, step=0.1, label="Ptychographic noise filter threshhold")
    if mode_selector.value == "SHG ptychographic":
        mo.output.append(mo.md("---"))
        mo.output.append(mo.md("### Ptychographic FROG options:"))
        if help_cb.value:
            mo.output.append(mo.md("Ptychographic reconstruction has additional options that may be configured. The ROI (region of interest) should include the reliably-measured part of the spectrogram. Bad sections of data, e.g. from the stitching region between spectrometers, may additionally be excluded. The noise threshold is an adjustable parameter, but likely won't benefit from adjustment."))
        mo.output.append(ptycho_roi_lower)
        mo.output.append(ptycho_roi_upper)
        mo.output.append(ptycho_exclude_lower)
        mo.output.append(ptycho_exclude_upper)
        mo.output.append(ptycho_threshhold)
    return (
        ptycho_exclude_lower,
        ptycho_exclude_upper,
        ptycho_roi_lower,
        ptycho_roi_upper,
        ptycho_threshhold,
    )


@app.cell
def _(help_cb, mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Optional spectral constraint:"))
    if help_cb.value:
        mo.output.append(mo.md("If available, a measured spectrum of the pulse to be reconstructed can be used to adjust the marginals of the measurement, improving agreement with the true spectrum."))
    spectral_constraint_file = mo.ui.file(label="Spectral contstraint file")
    mo.output.append(spectral_constraint_file)
    return (spectral_constraint_file,)


@app.cell
def _(aw, mo, spectral_constraint_file):
    spectral_constraint_format = mo.ui.dropdown(options=["Columns", "Text with headers"], value="Text with headers")
    spectral_constraint_data = spectral_constraint_file.contents()
    constraint_calibration_selector = mo.ui.dropdown(options=[e.value for e in aw.spectrum.CalibrationData],label="Calibration:")
    if spectral_constraint_data is not None:
        mo.output.append(spectral_constraint_format)
        mo.output.append(constraint_calibration_selector)
    return (
        constraint_calibration_selector,
        spectral_constraint_data,
        spectral_constraint_format,
    )


@app.cell
def _(mo, spectral_constraint_data, spectral_constraint_format):
    spectral_constraint_wavelength_header = mo.ui.text(label="Wavelength column key:", value="wavelength (nm)")
    spectral_constraint_wavelength_multiplier = mo.ui.number(label="Wavelength multiplier:", value=1e9)
    spectral_constraint_intensity_header = mo.ui.text(label="Intensity column key:", value="intensity (a.u.)")
    spectral_constraint_skip_lines = mo.ui.number(value=0, label="Header lines:")
    spectral_constraint_bandpass_f0 = mo.ui.number(value=375, label="Bandpass central frequency (THz)")
    spectral_constraint_bandpass_sigma = mo.ui.number(value=50, label="Bandpass width (THz)")
    spectral_constraint_bandpass_order = mo.ui.number(value=4, start=2, step=2, label="Bandpass order")
    if spectral_constraint_data is not None:
        if spectral_constraint_format.value == "Text with headers":
            mo.output.append(spectral_constraint_wavelength_header)
            mo.output.append(spectral_constraint_wavelength_multiplier)
            mo.output.append(spectral_constraint_intensity_header)
        if spectral_constraint_format.value == "Columns":
            mo.output.append(spectral_constraint_skip_lines)
        mo.output.append(spectral_constraint_bandpass_f0)
        mo.output.append(spectral_constraint_bandpass_sigma)
        mo.output.append(spectral_constraint_bandpass_order)
    return (
        spectral_constraint_bandpass_f0,
        spectral_constraint_bandpass_order,
        spectral_constraint_bandpass_sigma,
        spectral_constraint_intensity_header,
        spectral_constraint_skip_lines,
        spectral_constraint_wavelength_header,
        spectral_constraint_wavelength_multiplier,
    )


@app.cell
def _(
    aw,
    constraint_calibration_selector,
    spectral_constraint_bandpass_f0,
    spectral_constraint_bandpass_order,
    spectral_constraint_bandpass_sigma,
    spectral_constraint_data,
    spectral_constraint_format,
    spectral_constraint_intensity_header,
    spectral_constraint_skip_lines,
    spectral_constraint_wavelength_header,
    spectral_constraint_wavelength_multiplier,
):
    spectral_constraint = None
    if spectral_constraint_data is not None:
        match spectral_constraint_format.value:
            case "Columns":
                spectral_constraint = aw.data.load_mean_spectrum_from_scarab(
                    spectral_constraint_data.decode("utf-8"), is_data_string=True, header_size=spectral_constraint_skip_lines.value
                )
            case "Text with headers":
                spectral_constraint = aw.data.load_spectrum_from_text(
                    filename_or_data_string=spectral_constraint_data.decode("utf-8"),
                    wavelength_multiplier=1.0
                    / spectral_constraint_wavelength_multiplier.value,
                    wavelength_field=spectral_constraint_wavelength_header.value,
                    spectrum_field=spectral_constraint_intensity_header.value,
                    is_data_string=True
                )
        if constraint_calibration_selector.value is not None:
            constraint_calibration = aw.data.SpectrometerCalibration.from_npz(
                aw.spectrum.get_calibration_path() / constraint_calibration_selector.value
            )
            spectral_constraint = constraint_calibration.apply_to_spectrum(spectral_constraint)
        spectral_constraint = spectral_constraint.to_bandpassed(spectral_constraint_bandpass_f0.value * 1e12,spectral_constraint_bandpass_sigma.value * 1e12,int(spectral_constraint_bandpass_order.value))
    return (spectral_constraint,)


@app.cell
def _(aw, mo, plot_style_selector, spectral_constraint):
    if spectral_constraint is not None:
        mo.output.append(mo.md("### Loaded spectral constraint:"))

        if plot_style_selector.value == "Light":
            aw.plot.set_style("light")
            spectral_constraint.plot_with_group_delay()
            aw.plot.showmo()
        else:
            aw.plot.set_style("nick_dark")
            spectral_constraint.plot_with_group_delay()
            aw.plot.showmo()
    return


@app.cell
def _(help_cb, mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Run the reconstruction:"))
    if help_cb.value:
        mo.output.append(mo.md("The reconstruction will be run with multiple initial guesses to the spectral phase, with a given number of iterations (trial iterations) following each initial guess. After this, the guess which yielded the lowest error will be sent for additional iterations, set by the finishing iterations parameter. The trials will be run in multiple threads when run in a standard python environment, but the browser-based trials will not, due to current constraints on web assembly."))
    return


@app.cell
def _(is_in_web_notebook, mo):
    recon_trials = mo.ui.number(value=256, label="Initial guesses")
    recon_trial_length = mo.ui.number(value=16, label="Trial iterations")
    recon_followups = mo.ui.number(value=512, label="Finishing iterations")
    reconstruct_button = mo.ui.run_button(label="reconstruct")
    save_button = mo.ui.run_button(label="save")
    save_plot_button = mo.ui.run_button(label="save plot")
    mo.output.append(recon_trials)
    mo.output.append(recon_trial_length)
    mo.output.append(recon_followups)
    mo.output.append(reconstruct_button)

    if not is_in_web_notebook:
        mo.output.append(save_button)
        mo.output.append(save_plot_button)
    return (
        recon_followups,
        recon_trial_length,
        recon_trials,
        reconstruct_button,
        save_button,
        save_plot_button,
    )


@app.cell
def _(is_in_web_notebook, mo):
    if is_in_web_notebook:
        file_base = mo.ui.text(value="output", label="Output name")
        mo.output.append(file_base)
    return (file_base,)


@app.cell
def _(np):
    def resolve_frequency_roi(
        freq,
        start_roi: float,
        stop_roi: float,
        start_excluded_band: float,
        stop_excluded_band: float,
    ):
        roi = (freq >= start_roi) * (freq <= stop_roi)
        for i in range(len(freq)):
            if (freq[i] >= start_excluded_band) and (
                freq[i] <= stop_excluded_band
            ):
                roi[i] = False
        return np.fft.fftshift(roi)

    return (resolve_frequency_roi,)


@app.cell
def _(
    aw,
    frog_data,
    mo,
    mode_selector,
    ptycho_exclude_lower,
    ptycho_exclude_upper,
    ptycho_roi_lower,
    ptycho_roi_upper,
    ptycho_threshhold,
    recon_followups,
    recon_trial_length,
    recon_trials,
    reconstruct_button,
    resolve_frequency_roi,
    spectral_constraint,
    time,
    xfrog_reference,
):
    frog_type = None
    mo.stop(not reconstruct_button.value)
    if frog_data is not None:
        roi = None
        ptycho_threshhold_float = None
        match mode_selector.value:
            case "SHG GP":
                frog_type = aw.attoworld_rs.FrogType.Shg
            case "THG":
                frog_type = aw.attoworld_rs.FrogType.Thg
            case "Kerr":
                frog_type = aw.attoworld_rs.FrogType.Kerr
            case "XFROG":
                frog_type = aw.attoworld_rs.FrogType.Xfrog
            case "BlindFROG":
                frog_type = aw.attoworld_rs.FrogType.Blindfrog
            case "SHG ptychographic":
                frog_type = aw.attoworld_rs.FrogType.PtychographicShg
                roi = resolve_frequency_roi(
                    frog_data.freq,
                    ptycho_roi_lower.value * 1e12,
                    ptycho_roi_upper.value * 1e12,
                    ptycho_exclude_lower.value * 1e12,
                    ptycho_exclude_upper.value * 1e12,
                )
                ptycho_threshhold_float = ptycho_threshhold.value
        _start_time = time.time()
        result, result_gate, best_trial_index, best_finishing_index = aw.wave.reconstruct_frog(
            measurement=frog_data,
            repeats=int(recon_trials.value),
            test_iterations=int(recon_trial_length.value),
            polish_iterations=int(recon_followups.value),
            frog_type=frog_type,
            spectrum=spectral_constraint,
            xfrog_gate=xfrog_reference,
            roi=roi,
            ptychographic_threshhold=ptycho_threshhold_float,
        )
        _stop_time = time.time()
        reconstruction_time = _stop_time - _start_time
    return (
        best_finishing_index,
        best_trial_index,
        reconstruction_time,
        result,
        result_gate,
    )


@app.cell
def _(
    aw,
    display_download_link_from_file,
    file_base,
    is_in_web_notebook,
    mo,
    mode_selector,
    plot_style_selector,
    result,
    result_gate,
):
    if result is not None:
        if plot_style_selector.value == "Light":
            aw.plot.set_style("light")
            plot = result.plot_all(figsize=(9.6, 6), wavelength_autoscale=1e-3)
        else:
            aw.plot.set_style("nick_dark")
            plot = result.plot_all(figsize=(9.6, 6), wavelength_autoscale=1e-3)

        aw.plot.showmo()
        if mode_selector.value == "BlindFROG":
            mo.output.append(mo.md("### Gate"))
            plot_gate = result_gate.plot_all(
                figsize=(9.6, 6), wavelength_autoscale=1e-3
            )
            aw.plot.showmo()

        if is_in_web_notebook:
            plot.savefig("temp.svg")
            display_download_link_from_file(
                "temp.svg", output_name=f"{file_base.value}.svg"
            )
    return (plot,)


@app.cell
def _(help_cb, mo):
    if help_cb.value:
        mo.output.append(mo.md("**a** The measured spectrogram as binned above. **b** The retrieved spectrogram. **c** The reconstructed pulse, the time-dependent frequency as determined the derivative of the temporal phase, and the transform limited pulse. The Fourier limit is calculated with an amplitude gate set to $3 \\times 10^{-2}$ of the amplitude, or approximately three orders of magnitude in spectral intensity. **d** The retrieved spectrum and group delay curve."))
    return


@app.cell
def _(best_finishing_index, best_trial_index, mo, reconstruction_time, result):
    if result is not None:
        mo.output.append(mo.md(f"Reconstruction time: {reconstruction_time: .1f} s"))
        mo.output.append(mo.md(f"Best trial minimum error on iteration: {best_trial_index}"))
        mo.output.append(mo.md(f"Finishing iteration with lowest error: {best_finishing_index}"))
    return


@app.cell
def _(display_download_link_from_file, file_base, is_in_web_notebook, result):
    if (result is not None) and is_in_web_notebook:
        result.save_yaml(f"{file_base.value}.yml")
        display_download_link_from_file(
            f"{file_base.value}.yml",
            output_name=f"{file_base.value}.yml",
            mime_type="text/yaml",
        )
    return


@app.cell
def _(
    display_download_link_from_file,
    file_base,
    is_in_web_notebook,
    result,
    zipfile,
):
    if (result is not None) and is_in_web_notebook:
        result.save(file_base.value)
        with zipfile.ZipFile(f"{file_base.value}.zip", "w") as zip:
            zip.write(f"{file_base.value}.A.dat")
            zip.write(f"{file_base.value}.Arecon.dat")
            zip.write(f"{file_base.value}.Ek.dat")
            zip.write(f"{file_base.value}.Speck.dat")
            zip.write(f"{file_base.value}.yml")
            zip.write(f"{file_base.value}_negative_dazzler_phase.txt")
            zip.write(f"{file_base.value}_positive_dazzler_phase.txt")
        display_download_link_from_file(
            f"{file_base.value}.zip",
            output_name=f"{file_base.value}.zip",
            mime_type="application/zip",
        )
    return


@app.cell
def _(QFileDialog, is_in_web_notebook, mo, plot, result, save_button):
    mo.stop(not save_button.value)
    if not is_in_web_notebook:
        _file_path, _file_type = QFileDialog.getSaveFileName(
                None, "Save file", "", "All Files (*)")
        if (_file_path is not None) and (result is not None) and (_file_path != ""):
            result.save(_file_path)
            result.save_yaml(_file_path + ".yml")
            plot.savefig(_file_path + ".svg")
    return


@app.cell
def _(
    QFileDialog,
    is_in_web_notebook,
    mo,
    pathlib,
    plot,
    result,
    save_plot_button,
):
    mo.stop(not save_plot_button.value)
    if not is_in_web_notebook:
        _file_path, _file_type = QFileDialog.getSaveFileName(
                None, "Save file", "", "SVG files (*.svg);;PDF files (*.pdf)", "SVG files (*.svg)")
        if _file_path is not "" and result is not None:
            if pathlib.Path(_file_path).suffix is "":
                if _file_type == "PDF files (*.pdf)":
                    _file_path += ".pdf"
                else:
                    _file_path += ".svg"
            plot.savefig(_file_path)
    return


@app.cell
def _(help_cb, is_in_web_notebook, mo):
    dazzer_time_constant_box = mo.ui.number(value=100, step=1, start=0, label="Filter time constant (fs)")
    dazzler_filter_order_box = mo.ui.number(value=4, step=2, start=2, label="Filter order")
    dazzler_roi_min_box = mo.ui.number(value=700, start=100, step=1, label="ROI min (nm)")
    dazzler_roi_max_box = mo.ui.number(value=900, start=100, step=1, label="ROI max (nm)")
    dazzler_save_button = mo.ui.run_button(label="Save custom phase files")
    dazzler_old_phase_button = mo.ui.file(label="Load previous dazzler phase.txt")
    mo.output.append(mo.md("### Dazzler phase filtering:"))
    if help_cb.value:
        mo.output.append(mo.md("If the output phase will be fed-back to a Dazzler or other pulse-shaper, it is often useful to apply some smoothing to remove noise from the phase curve. Here, a supergaussian gate is applied to the group-delay curve. Shorter time constants yield more smoothing. The ROI only affects the plot."))
    mo.output.append(dazzer_time_constant_box)
    mo.output.append(dazzler_filter_order_box)
    mo.output.append(dazzler_roi_min_box)
    mo.output.append(dazzler_roi_max_box)
    mo.output.append(dazzler_old_phase_button)
    if not is_in_web_notebook:
        mo.output.append(dazzler_save_button)
    return (
        dazzer_time_constant_box,
        dazzler_filter_order_box,
        dazzler_old_phase_button,
        dazzler_roi_max_box,
        dazzler_roi_min_box,
        dazzler_save_button,
    )


@app.cell
def _(
    aw,
    dazzer_time_constant_box,
    dazzler_filter_order_box,
    dazzler_old_phase_button,
    dazzler_roi_max_box,
    dazzler_roi_min_box,
    display_download_link_from_file,
    is_in_web_notebook,
    np,
    plt,
    result,
):
    def phase_roi(phase_data, lam_min, lam_max):
        return phase_data[
            (phase_data[:, 0] <= lam_max) & (phase_data[:, 0] >= lam_min), :
        ]


    if result is not None:
        dazzler_phase_0 = phase_roi(
            result.spectrum.to_dazzer_phase(
                multiplier=1, filter_width=1.0, filter_order=8
            ),
            dazzler_roi_min_box.value,
            dazzler_roi_max_box.value,
        )
        dazzler_phase_filtered = phase_roi(
            result.spectrum.to_dazzer_phase(
                multiplier=1, filter_width=100e-15, filter_order=8
            ),
            dazzler_roi_min_box.value,
            dazzler_roi_max_box.value,
        )
        dazzler_phase_custom_filtered = result.spectrum.to_dazzer_phase(
            multiplier=1,
            filter_width=1e-15 * dazzer_time_constant_box.value,
            filter_order=dazzler_filter_order_box.value,
        )
        dazzler_phase_custom_filtered_negative = result.spectrum.to_dazzer_phase(
            multiplier=-1,
            filter_width=1e-15 * dazzer_time_constant_box.value,
            filter_order=dazzler_filter_order_box.value,
        )

        dazzler_old_phase_stream = dazzler_old_phase_button.contents()
        if dazzler_old_phase_stream is not None:
            dazzler_phase_old = aw.data.load_mean_spectrum_from_scarab(
                dazzler_old_phase_stream.decode("utf-8"), is_data_string=True, separator="\t"
            )
            interpolated_old_phase = aw.numeric.interpolate(
                np.array(dazzler_phase_custom_filtered[:, 0]),
                np.array(dazzler_phase_old.wavelength_nm()),
                np.array(dazzler_phase_old.spectrum),
            )
            dazzler_phase_custom_filtered[:, 1] += interpolated_old_phase
            dazzler_phase_custom_filtered_negative[:, 1] += interpolated_old_phase


        dazzler_phase_custom_filtered_roi = phase_roi(
            dazzler_phase_custom_filtered,
            dazzler_roi_min_box.value,
            dazzler_roi_max_box.value,
        )

        plt.plot(dazzler_phase_0[:, 0], dazzler_phase_0[:, 1], label="Unfiltered")
        plt.plot(
            dazzler_phase_filtered[:, 0],
            dazzler_phase_filtered[:, 1],
            label="Default filter",
        )
        if dazzler_old_phase_stream is None:
            plt.plot(
                dazzler_phase_custom_filtered_roi[:, 0],
                dazzler_phase_custom_filtered_roi[:, 1],
                label="Adjusted filter",
            )
        else:
            plt.plot(
                dazzler_phase_custom_filtered_roi[:, 0],
                dazzler_phase_custom_filtered_roi[:, 1],
                label="Adjusted + old phase",
            )
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Phase (rad)")
        aw.plot.showmo()

        if is_in_web_notebook:
            np.savetxt(
                "positive_phase.txt",
                dazzler_phase_custom_filtered,
                delimiter="\t",
                fmt="%12.12f",
                newline="\r\n",
            )
            np.savetxt(
                "negative_phase.txt",
                dazzler_phase_custom_filtered_negative,
                delimiter="\t",
                fmt="%12.12f",
                newline="\r\n",
            )
            display_download_link_from_file(
                "positive_phase.txt", output_name=f"positive_phase.txt"
            )
            display_download_link_from_file(
                "negative_phase.txt", output_name=f"negative_phase.txt"
            )
    return (
        dazzler_phase_custom_filtered,
        dazzler_phase_custom_filtered_negative,
    )


@app.cell
def _(
    QFileDialog,
    dazzler_phase_custom_filtered,
    dazzler_phase_custom_filtered_negative,
    dazzler_save_button,
    mo,
    np,
    result,
):
    mo.stop(not dazzler_save_button.value)
    _file_path, _file_type = QFileDialog.getSaveFileName(
            None, "Save file", "", "Base name (*)")
    if _file_path is not "" and result is not None:
        np.savetxt(_file_path+".txt",dazzler_phase_custom_filtered, delimiter="\t",
                fmt="%12.12f",
                newline="\r\n")
        np.savetxt(_file_path+"_negative.txt",dazzler_phase_custom_filtered_negative, delimiter="\t",
                fmt="%12.12f",
                newline="\r\n")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
