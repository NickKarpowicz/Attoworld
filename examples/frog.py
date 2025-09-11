# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
async def _():
    import marimo as mo

    # check if running in a browser for extra setup
    import sys
    is_in_web_notebook = sys.platform == "emscripten"
    if is_in_web_notebook:
        import micropip
        import os
        import zipfile
        await micropip.install(
            "https://nickkarpowicz.github.io/wheels/attoworld-2025.0.42-cp312-cp312-emscripten_3_1_58_wasm32.whl"
        )
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
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

    import attoworld as aw
    import numpy as np
    import pathlib

    aw.plot.set_style("nick_dark")
    return (
        aw,
        display_download_link_from_file,
        filedialog,
        is_in_web_notebook,
        mo,
        np,
        pathlib,
        zipfile,
    )


@app.cell
def _(mo):
    mo.output.append(mo.md("# FROG Reconstruction:"))
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Select your FROG file:"))
    return


@app.cell
def _(mo):
    file_browser = mo.ui.file(filetypes=[".dwc"], label="Select .dwc file")
    file_browser
    return (file_browser,)


@app.cell
def _(aw, mo):
    calibration_selector = mo.ui.dropdown(options=[e.value for e in aw.spectrum.CalibrationData],label="Calibration:")
    calibration_selector
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
def _(aw, calibration_selector, file_browser):
    _path = file_browser.contents()
    if _path is not None:
        input_data = aw.data.read_dwc(file_or_path=_path, is_buffer=True)
        if calibration_selector.value is not None:
            calibration = aw.data.SpectrometerCalibration.from_npz(
                aw.spectrum.get_calibration_path() / calibration_selector.value
            )
            input_data = calibration.apply_to_spectrogram(input_data)
    else:
        input_data = None
    return (input_data,)


@app.cell
def _(mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Bin data onto evenly spaced grid:"))
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
    aw,
    bin_settings,
    display_download_link_from_file,
    input_data,
    is_in_web_notebook,
):
    # if not bin_live.value:
    #     mo.stop(not bin_button.value)
    if input_data is not None:
        frog_data = input_data.to_bin_pipeline_result(bin_settings)
        frog_data.plot_log()
        aw.plot.showmo()

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
def _(mo, mode_selector):
    ptycho_roi_lower = mo.ui.number(value = 300.0, step=0.1, label="ROI lower frequency (THz)")
    ptycho_roi_upper = mo.ui.number(value = 900.0, step=0.1, label="ROI upper frequency (THz)")
    ptycho_exclude_lower = mo.ui.number(value = 1000.0, step=0.1, label="Excluded region lower frequency (THz)")
    ptycho_exclude_upper = mo.ui.number(value = 1200.0, step=0.1, label="Excluded region upper frequency (THz)")
    ptycho_threshhold = mo.ui.number(value = 10000.0, step=0.1, label="Ptychographic noise filter threshhold")
    if mode_selector.value == "SHG ptychographic":
        mo.output.append(mo.md("---"))
        mo.output.append(mo.md("### Ptychographic FROG options:"))
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
def _(mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Optional spectral constraint:"))
    spectral_constraint_file = mo.ui.file(label="Spectral contstraint file")
    mo.output.append(spectral_constraint_file)
    return (spectral_constraint_file,)


@app.cell
def _(mo, spectral_constraint_file):
    spectral_constraint_format = mo.ui.dropdown(options=["Columns", "Text with headers"], value="Text with headers")
    spectral_constraint_data = spectral_constraint_file.contents()
    if spectral_constraint_data is not None:
        mo.output.append(spectral_constraint_format)
    return spectral_constraint_data, spectral_constraint_format


@app.cell
def _(mo, spectral_constraint_data, spectral_constraint_format):
    spectral_constraint_wavelength_header = mo.ui.text(label="Wavelength column key:", value="wavelength (nm)")
    spectral_constraint_wavelength_multiplier = mo.ui.number(label="Wavelength multiplier:", value=1e9)
    spectral_constraint_intensity_header = mo.ui.text(label="Intensity column key:", value="intensity (a.u.)")
    spectral_constraint_skip_lines = mo.ui.number(value=0, label="Header lines:")
    if spectral_constraint_data is not None:
        if spectral_constraint_format.value == "Text with headers":
            mo.output.append(spectral_constraint_wavelength_header)
            mo.output.append(spectral_constraint_wavelength_multiplier)
            mo.output.append(spectral_constraint_intensity_header)
        if spectral_constraint_format.value == "Columns":
            mo.output.append(spectral_constraint_skip_lines)
    return (
        spectral_constraint_intensity_header,
        spectral_constraint_skip_lines,
        spectral_constraint_wavelength_header,
        spectral_constraint_wavelength_multiplier,
    )


@app.cell
def _(
    aw,
    mo,
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
        mo.output.append(mo.md("### Loaded spectral constraint:"))
        spectral_constraint.plot_with_group_delay()
        aw.plot.showmo()
    return (spectral_constraint,)


@app.cell
def _(mo):
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### Run the reconstruction:"))
    return


@app.cell
def _(is_in_web_notebook, mo):
    recon_trials = mo.ui.number(value=16, label="Initial guesses")
    recon_trial_length = mo.ui.number(value=128, label="Trial iterations")
    recon_followups = mo.ui.number(value=1024, label="Finishing iterations")
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
    xfrog_reference,
):
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
        result, result_gate = aw.wave.reconstruct_frog(
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
    else:
        result = None
    return result, result_gate


@app.cell
def _(
    aw,
    display_download_link_from_file,
    file_base,
    is_in_web_notebook,
    mo,
    mode_selector,
    result,
    result_gate,
):
    if result is not None:
        plot = result.plot_all(figsize=(9, 6), wavelength_autoscale=1e-3)
        aw.plot.showmo()
        if mode_selector.value == "BlindFROG":
            mo.output.append(mo.md("### Gate"))
            plot_gate = result_gate.plot_all(
                figsize=(9, 6), wavelength_autoscale=1e-3
            )
            aw.plot.showmo()

        if is_in_web_notebook:
            plot.savefig("temp.svg")
            display_download_link_from_file(
                "temp.svg", output_name=f"{file_base.value}.svg"
            )
    return (plot,)


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
        display_download_link_from_file(
            f"{file_base.value}.zip",
            output_name=f"{file_base.value}.zip",
            mime_type="application/zip",
        )
    return


@app.cell
def _(filedialog, is_in_web_notebook, mo, result, save_button):
    mo.stop(not save_button.value)
    if not is_in_web_notebook:
        _file_path = filedialog.asksaveasfilename(
            title="Save File", filetypes=[("All Files", "*.*")]
        )

        if (_file_path is not None) and (result is not None) and (_file_path != ""):
            result.save(_file_path)
            result.save_yaml(_file_path + ".yml")
    return


@app.cell
def _(filedialog, is_in_web_notebook, mo, plot, result, save_plot_button):
    mo.stop(not save_plot_button.value)
    if not is_in_web_notebook:
        _file_path = filedialog.asksaveasfilename(
            title="Save File", filetypes=[("SVG files", "*.svg"), ("PDF files", "*.pdf")]
        )

        if _file_path is not None and result is not None:
            plot.savefig(_file_path)
    return


@app.cell
def _(bin_save_button, bin_settings, filedialog, is_in_web_notebook, mo):
    if not is_in_web_notebook:
        mo.stop(not bin_save_button.value)
        _file_path = filedialog.asksaveasfilename(
            title="Save File", filetypes=[("YAML files", "*.yml")]
        )

        if _file_path is not None and bin_settings is not None:
            bin_settings.save_yaml(_file_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
