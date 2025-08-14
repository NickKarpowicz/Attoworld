import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
async def _():
    import marimo as mo

    # check if running in a browser, install attoworld from local copy
    import sys

    is_in_web_notebook = sys.platform == "emscripten"
    if is_in_web_notebook:
        import micropip
        import os

        path_to_attoworld = (
            mo.notebook_location()
            / "public"
            / "attoworld-2025.0.38-cp312-cp312-emscripten_3_1_58_wasm32.whl"
        )
        micropip.uninstall("attoworld")
        await micropip.install(str(path_to_attoworld))

        import base64
        import zipfile

        def create_download_link(data, filename, mime_type="text/plain"):
            encoded_data = base64.b64encode(data).decode("utf-8")
            data_uri = f"data:{mime_type};base64,{encoded_data}"
            html = f'<a href="{data_uri}" download="{filename}">Download {filename}</a>'
            return mo.Html(html)

        def display_download_link_from_file(
            path, output_name, mime_type="text/plain"
        ):
            with open(path, "rb") as _file:
                mo.output.append(
                    create_download_link(
                        data=_file.read(),
                        filename=output_name,
                        mime_type=mime_type,
                    )
                )
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

    import attoworld as aw
    import numpy as np

    aw.plot.set_style("nick_dark")
    return (
        aw,
        display_download_link_from_file,
        filedialog,
        is_in_web_notebook,
        mo,
        np,
        zipfile,
    )


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
def _(aw, bin_spatial_chirp_correction, calibration_selector, file_browser):
    _path = file_browser.contents()
    if _path is not None:
        input_data = aw.data.read_dwc(file_or_path=_path, is_buffer=True)
        if calibration_selector.value is not None:
            calibration = aw.data.SpectrometerCalibration.from_npz(
                aw.spectrum.get_calibration_path() / calibration_selector.value
            )
            input_data = calibration.apply_to_spectrogram(input_data)
        if bin_spatial_chirp_correction.value:
            input_data.to_removed_spatial_chirp()
    else:
        input_data = None
    return (input_data,)


@app.cell
def _(mo):
    bin_size = mo.ui.number(label="size", value=96, step=2)
    bin_dt = mo.ui.number(label="dt (fs)", value=3, step=0.1)
    bin_f0 = mo.ui.number(label="f0 (THz)", value=740, step=1)
    bin_offset = mo.ui.number(label="dark noise level", value=0.0002, step=1e-5)
    bin_fblock = mo.ui.number(label="freq block avg.", value=16, step=1)
    bin_tblock = mo.ui.number(label="time block avg.", value=1, step=1)
    bin_median = mo.ui.checkbox(label="median blocking", value=False)
    bin_spatial_chirp_correction = mo.ui.checkbox(label="correct spatial chirp", value=False)
    bin_button = mo.ui.run_button(label="bin")
    bin_live = mo.ui.checkbox(label="live update")
    mo.output.append(bin_size)
    mo.output.append(bin_dt)
    mo.output.append(bin_f0)
    mo.output.append(bin_offset)
    mo.output.append(bin_fblock)
    mo.output.append(bin_tblock)
    mo.output.append(bin_median)
    mo.output.append(bin_spatial_chirp_correction)
    mo.output.append(bin_live)
    mo.output.append(bin_button)

    return (
        bin_button,
        bin_dt,
        bin_f0,
        bin_fblock,
        bin_live,
        bin_median,
        bin_offset,
        bin_size,
        bin_spatial_chirp_correction,
        bin_tblock,
    )


@app.cell
def _(
    aw,
    bin_button,
    bin_dt,
    bin_f0,
    bin_fblock,
    bin_live,
    bin_median,
    bin_offset,
    bin_size,
    bin_tblock,
    input_data,
    mo,
):
    if not bin_live.value:
        mo.stop(not bin_button.value)
    if input_data is not None:
        if bin_median.value:
            _method = "median"
        else:
            _method = "mean"
        frog_data = (
            input_data.to_block_binned(
                int(bin_fblock.value), int(bin_tblock.value), method=_method
            )
            .to_binned(
                dim=int(bin_size.value),
                dt=float(bin_dt.value * 1e-15),
                f0=float(bin_f0.value * 1e12),
            )
            .to_per_frequency_dc_removed(extra_offset=float(bin_offset.value))
        )
        frog_data.plot_log()
        aw.plot.showmo()
    else:
        frog_data = None
    return (frog_data,)


@app.cell
def _(is_in_web_notebook, mo):
    recon_trials = mo.ui.number(value=8, label="Initial guesses")
    recon_trial_length = mo.ui.number(value=64, label="Trial iterations")
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
def _(
    aw,
    frog_data,
    mo,
    recon_followups,
    recon_trial_length,
    recon_trials,
    reconstruct_button,
):
    mo.stop(not reconstruct_button.value)
    if frog_data is not None:
        result = aw.wave.reconstruct_shg_frog(
            measurement=frog_data,
            repeats=int(recon_trials.value),
            test_iterations=int(recon_trial_length.value),
            polish_iterations=int(recon_followups.value),
        )
    else:
        result = None
    return (result,)


@app.cell
def _(
    aw,
    display_download_link_from_file,
    file_base,
    is_in_web_notebook,
    np,
    result,
    zipfile,
):
    if result is not None:
        spec = result.spectrum.to_intensity_spectrum()
        indices = np.where(spec.spectrum / np.max(spec.spectrum) > 3e-3)[0]
        wl_nm = spec.wavelength_nm()
        plot = result.plot_all(
            figsize=(9, 6),
            wavelength_xlims=(wl_nm[indices[-1]], wl_nm[indices[0]]),
        )
        aw.plot.showmo()

        if is_in_web_notebook:
            plot.savefig("temp.svg")
            display_download_link_from_file(
                "temp.svg", output_name=f"{file_base.value}.svg"
            )

            result.save(file_base.value)
            result.save_yaml(f"{file_base.value}.yaml")
            with zipfile.ZipFile(f"{file_base.value}.zip", "w") as zip:
                zip.write(f"{file_base.value}.A.dat")
                zip.write(f"{file_base.value}.Arecon.dat")
                zip.write(f"{file_base.value}.Ek.dat")
                zip.write(f"{file_base.value}.Speck.dat")
                zip.write(f"{file_base.value}.yaml")
            display_download_link_from_file(
                f"{file_base.value}.zip",
                output_name=f"{file_base.value}.zip",
                mime_type="application/zip",
            )
    return (plot,)


@app.cell
def _(filedialog, is_in_web_notebook, mo, result, save_button):
    mo.stop(not save_button.value)
    if not is_in_web_notebook:
        _file_path = filedialog.asksaveasfilename(
            title="Save File", filetypes=[("All Files", "*.*")]
        )

        if (_file_path is not None) and (result is not None) and (_file_path != ""):
            result.save(_file_path)
            result.save_yaml(_file_path + ".yaml")
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
def _():
    return


if __name__ == "__main__":
    app.run()
