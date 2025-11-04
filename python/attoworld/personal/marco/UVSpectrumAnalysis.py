import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# load calibration data
def load_calibration_data(calibration_data_filepath):
    """Load calibration data in a .npz file for the Reso spectrometer.

    Args:
        calibration_data_filepath (StrOrBytesPath): path of the file to load

    """
    try:
        calibration = np.load(calibration_data_filepath)
        wavelength_calibration = calibration["wavelength"]
        lamp_spec = calibration["lamp_ref"]
        lamp_measbyReso = calibration["lamp_measured"]
        calibration_smoothed = np.abs(calibration["corr_factor_smoothed"])
    except FileNotFoundError:
        print(
            "Error: calibration data for the UV spectrometer not found.\n"
            "Please copy the folder Attoworld/src/attoworld/spectrum/calibration_data into your current working directory\n"
            "or alternatively create a calibration file with relative path ./calibration_data/Reso_Spectrometer_CalibrationCorrection.npz\n"
        )
        raise FileNotFoundError("calibration data not found")
    return wavelength_calibration, lamp_spec, lamp_measbyReso, calibration_smoothed


def smooth(y, box_pts: int):
    """Basic box smoothing for the spectra.

    Arguments:
        y = spectrum to be smoothed [numpy array]
        box_pts = width of smoothing box [int, number of points]

    Returns:
        smoothed_y [numpy array]

    """
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def tukey_f(x: float, center: float, FWHM: float, w: float):
    """cosine-smoothed, flattop window function centered in 'center'.

    each edge is w-wide
    amplitude = 1

    Arguments:
        x: float = input value
        center = center of the t. function
        FWHM = full width at half maximum of the t. function
        w = width of the edges of the t. function (0 < w < FWHM)

    """
    if FWHM < w:
        print("tukey window can not have edges wider than FWHM")
    xmin = center - FWHM / 2 - w / 2
    xmax = center + FWHM / 2 + w / 2
    if x < xmin or x > xmax:
        y = 0
    elif xmin <= x and x < xmin + w:
        y = (1 - np.cos((x - xmin) * np.pi / w)) / 2
    elif xmin + w <= x and x <= xmax - w:
        y = 1
    elif xmax - w < x and x <= xmax:
        y = (1 - np.cos((x - xmax) * np.pi / w)) / 2
    else:
        print(x, xmax, xmin, w)
        raise ValueError(
            "in tukey_f, x could not be assigned correctly to the sub intervals, might it (or the other parameters) be NaN?"
        )
    return y


def tukey_window(x, center: float, FWHM: float, w: float):
    """Returns a cosine-smoothed, flattop window centered in 'center'; amplitude = 1.

    The difference with the function tukey_f is that this function can take as input a numpy array or a list, and returns a numpy array or a list.

    Args:
        x: input array (numpy array or list)
        center: center of the window
        FWHM: full width at half maximum
        w: width of the edges of the window (w > 0)

    Returns:
        window (numpy array or list)

    """
    if isinstance(x, np.ndarray):
        y = []
        for xi in x:
            y.append(tukey_f(xi, center, FWHM, w))
        y = np.array(y)
    elif isinstance(x, list):
        y = []
        for xi in x:
            y.append(tukey_f(xi, center, FWHM, w))
    elif isinstance(x, float):
        y = tukey_f(x, center, FWHM, w)
    else:
        raise TypeError(
            "in function tukey window x is neither np array nor list nor float"
        )
    return y


def read_csd_file(filename):
    """DEPRECATED read the file produced by the Maya spectrometer."""
    file = open(filename)
    data = []
    for row in file:
        # row = row.replace(',', '.')
        elemlist = str(row).split()
        row_float = []
        for elem in elemlist:
            row_float.append(float(elem))
        data.append(row_float)
    file.close()
    return np.array(data)


def plot_spectra(filenameList, pdfFilename, legendItemList=None):
    """DEPRECATED plot the spectra produced by the Maya spectrometer."""
    dataList = []
    for filename in filenameList:
        dataList.append(read_csd_file(filename))

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    legendHandles = []
    for data in dataList:
        avgeSpectrum = []
        for row in data:
            avgeSpectrum.append(np.mean(np.array(row)))
        (handle,) = ax[0].plot(data[:, 0], avgeSpectrum)
        legendHandles.append(handle)
    # ax[0].set_ylim(3e-17,1e-9)
    # ax[0].set_xlim(freqLim[0], freqLim[1])
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].set_ylabel("Intensity")
    if legendItemList is not None:
        ax[0].legend(legendHandles, legendItemList)

    for data in dataList:
        avgeSpectrum = []
        for row in data:
            avgeSpectrum.append(np.mean(np.array(row)))
        (handle,) = ax[1].semilogy(data[:, 0], avgeSpectrum)
        legendHandles.append(handle)
    # ax[1].set_ylim(3e-17,1e-9)
    # ax[1].set_xlim(freqLim[0], freqLim[1])
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_ylabel("Intensity")
    if legendItemList is not None:
        ax[1].legend(legendHandles, legendItemList)

    fig.savefig(pdfFilename)


def read_spectrometer_excel(filename):
    """Reads xls file (passed as the string filename without extension!) produced by the UV-VIS spectrometer RESONANCE VS-7550-VUV.

    This function currently only works in Linux (Ubuntu), since it uses the OS system command to brutally copy the content of the xls file into a txt;
    such copying is necessary because the xls file is not readable by pandas (it's a non-standard xls, I couldn't find any other workaround).
    For windows there is a similar command, please replace the line
    os.system("cat " + filename + ".xls > " + filename + ".txt")

    44 (skipped) rows at the beginning of the file (all headers)

    Args:
        filename: name of the file (without extension)

    Returns:
        numpy array with the spectral data (see original excel for more details)

    """
    if "." in filename and "./" not in filename:
        raise ValueError(
            "in function read_spectrometer_excel, filename must be passed without extension (filename should not contain a dot)"
        )
    os.system("cat " + filename + ".xls > " + filename + ".txt")
    dataF = pd.read_table(
        filename + ".txt", sep="\t", keep_default_na=False, skiprows=44
    )  # keep_default_na=False,
    data = []
    for row in dataF.values:
        data.append([])
        for x in row:
            if x == "":
                data[-1].append(float("nan"))
            else:
                data[-1].append(float(x))
    return np.array(data)


def calibrate(
    data,
    column: int,
    calibration_file_path="./calibration_data/Reso_Spectrometer_CalibrationCorrection.npz",
    dark=None,
    dark_c=None,
    stitch: bool = False,
    smooth_points: int = 10,
    null_calibration: bool = False,
    wavelength_calibration_intercept: float = 3.538,
    wavelength_calibration_slope: float = 1.003427,
):
    """For the UV spectrometer. Calibrate the spectrum number 'column' in the the 'data' array. Intensity calibration factor is loaded in the first lines of UVSpectrumAnalysis.py.
        Notice: each saved spectrum has 7 columns in the excel file; however, the argument column refers to the index of the spectrum, not the actual column in the file.

    Args:
        data: data [numpy] array (see read_spectrometer_excel)
        column: index of the spectrum to be calibrated
        calibration_file_path: path of the calibration file
        dark: dark spectrum (optional)
        dark_c: column of the dark spectrum (optional)
        stitch: Boolean, if True, the spectra are stitched together (optional, default False)
                if True, the spectrum will be calculated by stitching the 5 partial spectra in the previous columns.
                the stitching only works if the spectrum number 'column' is a FULL-SPECTRUM in the xls file.
                Stitching was implemented because the internal software of the spectrometer is not good enough at it (the saved full spectrum typically has discontinuities).
        smooth_points: number of points for smoothing (optional, default 10). if 0, no smoothing is applied
        null_calibration: Boolean, if True, the intensity calibration is not applied (optional, default False)
        wavelength_calibration_intercept: intercept of the wavelength calibration (optional, default 3.538). If None no wavelength calibration is applied
        wavelength_calibration_slope: slope of the wavelength calibration (optional, default 1.003427). If None no wavelength calibration is applied
                Wavelength calibration is in the form λ_true = λ_measured * slope + intercept
                Notice: wavelength calibration has an error of +/- 1  to 2 nm due to the stitching process.
                    for a better calibration compute and apply the correction to each of the 5 partial spectra

    Returns:
        wavelength: wavelength array
        spectrum: calibrated spectrum

    """
    wavelength_calibration, lamp_spec, lamp_measbyReso, calibration_smoothed = (
        load_calibration_data(calibration_file_path)
    )

    if wavelength_calibration_slope is None or wavelength_calibration_intercept is None:
        cal_slope = 1.0
        cal_intercept = 0.0
    else:
        cal_slope = wavelength_calibration_slope
        cal_intercept = wavelength_calibration_intercept

    if stitch:
        # check that the chosen spectrum (in column) is actually a full one
        theoretically_complete_wvl = (
            data[:, 7 * (column) + 1] * cal_slope + cal_intercept
        )
        theoretically_complete_wvl = theoretically_complete_wvl[
            ~np.isnan(theoretically_complete_wvl)
        ]
        if theoretically_complete_wvl[-1] - theoretically_complete_wvl[0] < 900:
            raise ValueError(
                "in function calibrate(), the spectrum to be stitched is not a full spectrum, please check the column index"
            )

        extrema = []
        spectra = []
        for i in range(5):
            wvl_single_sp = (
                data[:, 7 * (column - 5 + i) + 1] * cal_slope + cal_intercept
            )
            single_sp = data[:, 7 * (column - 5 + i) + 2]
            if dark is not None and dark_c is not None:
                single_sp -= dark[:, 7 * (dark_c - 5 + i) + 2]
            wvl_single_sp = wvl_single_sp[~np.isnan(wvl_single_sp)]
            single_sp = single_sp[~np.isnan(single_sp)]
            extrema.append([wvl_single_sp[0], wvl_single_sp[-1]])
            spectra.append([wvl_single_sp, single_sp])

        overlaps = []
        centers_of_overlap = [extrema[0][0]]
        for i in range(4):
            overlaps.append(extrema[i][1] - extrema[i + 1][0])
            centers_of_overlap.append(extrema[i][1] / 2 + extrema[i + 1][0] / 2)

        w = np.min(overlaps)
        centers_of_overlap.append(extrema[4][1] + w / 2)
        centers_of_overlap[0] = centers_of_overlap[0] - w / 2

        wavelength = data[:, 7 * column + 1] * cal_slope + cal_intercept
        spectrum = wavelength * 0.0

        for i in range(5):
            center = centers_of_overlap[i] / 2 + centers_of_overlap[i + 1] / 2
            FWHM = centers_of_overlap[i + 1] - centers_of_overlap[i]
            spectrum += np.interp(
                wavelength,
                spectra[i][0],
                spectra[i][1] * tukey_window(spectra[i][0], center, FWHM, w),
            )
            plt.plot(
                spectra[i][0],
                spectra[i][1] * tukey_window(spectra[i][0], center, FWHM, w),
            )
        calibr_interpolated = np.interp(
            wavelength, wavelength_calibration, calibration_smoothed
        )
        if not null_calibration:
            spectrum *= calibr_interpolated

        plt.plot(
            data[:, 7 * column + 1] * cal_slope + cal_intercept, data[:, 7 * column + 2]
        )
        plt.plot(wavelength, spectrum)

        if w < 0:
            raise ValueError(
                "in function calibrate(), the spectra to be stitched do not overlap"
            )

    else:
        wavelength = data[:, 7 * column + 1] * cal_slope + cal_intercept
        if dark is None or dark_c is None:
            calibr_interpolated = np.interp(
                wavelength, wavelength_calibration, calibration_smoothed
            )
            if null_calibration:
                calibr_interpolated = calibr_interpolated * 0.0 + 1
            spectrum = (
                data[:, 7 * column + 2] - np.min(data[:, 7 * column + 2])
            ) * calibr_interpolated
        else:
            calibr_interpolated = np.interp(
                wavelength, wavelength_calibration, calibration_smoothed
            )
            if null_calibration:
                calibr_interpolated = calibr_interpolated * 0.0 + 1
            spectrum = (
                data[:, 7 * column + 2] - dark[:, 7 * dark_c + 2]
            ) * calibr_interpolated

            if len(data[:, 7 * column + 2][~np.isnan(data[:, 7 * column + 2])]) != len(
                dark[:, 7 * dark_c + 2][~np.isnan(dark[:, 7 * dark_c + 2])]
            ):
                raise ValueError(
                    "In function calibrate(), the dark spectrum is not the same length as the spectrum to be calibrated.\n"
                    "Please check the column indices of the dark spectrum and the main spectrum"
                )
            if (
                data[:, 7 * column + 1][~np.isnan(data[:, 7 * column + 1])][0]
                != dark[:, 7 * dark_c + 2][~np.isnan(dark[:, 7 * dark_c + 1])][0]
            ):
                print(
                    "WARNING: In function calibrate(), the dark spectrum does not have the same wavelength as the spectrum to be calibrated.\n"
                    "Please check the column indices of the dark spectrum and the main spectrum"
                )

    if smooth_points is not None and smooth_points > 0:
        spectrum = smooth(spectrum, smooth_points)
    return wavelength, spectrum


def list_error():
    raise Exception("list size in function plot_spectra_UVsp is not coherent")


def plot_spectra_UVsp(
    filenameList,
    columnList,
    pdfFilename,
    legendItemList=None,
    darkTupleList=None,
    normalizationList=None,
    color_gradient: bool = False,
    additive_constant_log_intensity=20,
    wavelength_range=None,
    title=None,
    invert_order: bool = False,
    plotList=None,
    do_calibrate: bool = True,
    stitch: bool = False,
    smooth_points: int = 10,
    calibration_file_path="./calibration_data/Reso_Spectrometer_CalibrationCorrection.npz",
):
    """Plotting function for the UV spectrometer.

    Notice: The excel file usually contains several spectra.
    Notice: each saved spectrum has 7 columns in the excel file; please consider that the argument 'column' refers to the index of the spectrum, not the actual column in the file.

    Args:
        filenameList: input file list (without extension)
        columnList: list of the positions in the file of the spectra to be plotted (e.g. [1, 0, 5] -> plot 1st sp. for 1st file, 5th spectrum for 3rd file, ...)
        pdfFilename: output filename
        darkTupleList: [("darkFilename1stspectrum", spectrum_number_in_1stDark_filename), ( ..,.. ), ...]
        legendItem List: list of entries for the legend (optional, keyword)
        normalizationList: list of (multiplicative) normalization factors for the spectra (optional, keyword)
        color_gradient: Boolean, whether plot should be displayed with color gradient instead of in matplotlib default color cycle
        additive_constant_log_intensity: additive constant to the spectral intensity, so that the logarithmic plot does not have values too close to 0
        wavelength_range: [min lambda, max lambda] for plotting
        title: title of the figure
        plotList: list of booleans: this function will plot the entry filenameList[i] if plotList[i] = True, otherwise the i-th data will not be shown in the plot
        invert_order: Boolean, if True, the order of the spectra will be inverted (i.e. the last spectrum in the list will be plotted first)
        do_calibrate: Boolean, if True, the spectra will be calibrated (default True, see calibrate() function)
        calibration_file_path: str, path to the calibration file (default "./calibration_data/Reso_Spectrometer_CalibrationCorrection.npz"). Ignored if do_calibrate = False
        stitch: Boolean, if True, the spectra will be stitched together (optional, default False)
            if True, the spectrum will be calculated by stitching the 5 partial spectra in the previous columns.
            the stitching only works if the spectrum number 'column' is a FULL-SPECTRUM in the xls file.
            Stitching was implemented because the internal software of the spectrometer does not do it properly (the saved full spectrum tipically has discontinuities).
            Stitching here works only if do_calibrate = True
        smooth_points: number of points for smoothing (optional, default 10)

    DEPENDS ON read_spectrometer_excel()
    DEPENDS ON calibrate() [optional]

    """
    if len(columnList) != len(filenameList):
        print(
            "Error: in function plot_spectra_UVsp\ncolumnList has different size than filenameList"
        )
        list_error()
    if legendItemList is not None:
        if len(legendItemList) != len(filenameList):
            print(
                "Error: in function plot_spectra_UVsp\nlegendItemList has different size than filenameList"
            )
            list_error()
    if darkTupleList is not None:
        if len(darkTupleList) != len(filenameList):
            print(
                "Error: in function plot_spectra_UVsp\ndarkTupleList has different size than filenameList"
            )
            list_error()
    if normalizationList is not None:
        if len(normalizationList) != len(filenameList):
            print(
                "Error: in function plot_spectra_UVsp\nnormalizationList has different size than filenameList"
            )
            list_error()
    if plotList is not None:
        if len(plotList) != len(filenameList):
            print(
                "Error: in function plot_spectra_UVsp\nplotList has different size than filenameList"
            )
            list_error()

    if invert_order:
        filenameList.reverse()
        columnList.reverse()
        if legendItemList is not None:
            legendItemList.reverse()
        if darkTupleList is not None:
            darkTupleList.reverse()
        if normalizationList is not None:
            normalizationList.reverse()
        if plotList is not None:
            plotList.reverse()

    if plotList is not None:
        for i in reversed(range(len(plotList))):
            if not plotList[i]:
                if filenameList is not None:
                    del filenameList[i]
                if columnList is not None:
                    del columnList[i]
                if legendItemList is not None:
                    del legendItemList[i]
                if darkTupleList is not None:
                    del darkTupleList[i]
                if normalizationList is not None:
                    del normalizationList[i]

    dataList = []
    for filename in filenameList:
        data = read_spectrometer_excel(filename)
        dataList.append(data)

    # check if the dark spectrum is shorter or longer than the spectrum to be calibrated (if it has nan values)
    # [when a = nan the statement (a == a) returns False]
    if darkTupleList is not None:
        for tp, data, column in zip(darkTupleList, dataList, columnList):
            dark = read_spectrometer_excel(tp[0])
            for i in range(len(data[:, 7 * column + 2])):
                if (
                    dark[i, 7 * tp[1] + 2] != dark[i, 7 * tp[1] + 2]
                    and data[i, 7 * column + 2] == data[i, 7 * column + 2]
                ):
                    print(
                        "Warning: in function plot_spectra_UVsp\ndark spectrum is shorter than the spectrum to be calibrated"
                        "\nplease check the column index of the dark spectrum"
                    )
                    break
                if (
                    data[i, 7 * column + 2] != data[i, 7 * column + 2]
                    and dark[i, 7 * tp[1] + 2] == dark[i, 7 * tp[1] + 2]
                ):
                    print(
                        "Warning: in function plot_spectra_UVsp\ndark spectrum is longer than the spectrum to be calibrated"
                        "\nplease check the column indices of the dark spectrum and the main spectrum"
                    )
                    break

    # subtract dark (if any, and if it will not be subtracted during calibration)
    if darkTupleList is not None and not do_calibrate:
        for tp, data, column in zip(darkTupleList, dataList, columnList):
            dark = read_spectrometer_excel(tp[0])
            for i in range(len(data[:, 7 * column + 2])):
                data[i, 7 * column + 2] = (
                    data[i, 7 * column + 2] - dark[i, 7 * tp[1] + 2]
                )

    if do_calibrate:
        for i in range(len(dataList)):
            if darkTupleList is not None:
                dark = read_spectrometer_excel(darkTupleList[i][0])
                (
                    dataList[i][:, 7 * columnList[i] + 1],
                    dataList[i][:, 7 * columnList[i] + 2],
                ) = calibrate(
                    dataList[i],
                    columnList[i],
                    calibration_file_path=calibration_file_path,
                    dark=dark,
                    dark_c=darkTupleList[i][1],
                    stitch=stitch,
                    smooth_points=smooth_points,
                )
            else:
                (
                    dataList[i][:, 7 * columnList[i] + 1],
                    dataList[i][:, 7 * columnList[i] + 2],
                ) = calibrate(
                    dataList[i],
                    columnList[i],
                    calibration_file_path=calibration_file_path,
                    calibration_dark=None,
                    dark_c=None,
                    stitch=stitch,
                    smooth_points=smooth_points,
                )

    n_lines = len(dataList)
    if color_gradient:
        # Take colors at regular intervals spanning the colormap.
        cmap = mpl.colormaps[
            "magma"
        ]  #'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        colors = cmap(np.linspace(0, 0.92, n_lines))
        # fig, ax = plt.subplots(layout='constrained')
    else:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = colors[:n_lines]

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    if title is not None:
        fig.suptitle(title)
    ax[0].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax[1].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    legendHandles = []
    if normalizationList is None:
        normalizationList = [1.0 for i in dataList]
    for data, column, normalization, color, i in zip(
        dataList, columnList, normalizationList, colors, range(len(dataList))
    ):
        (handle,) = ax[0].plot(
            data[:, 7 * column + 1],
            data[:, 7 * column + 2] * normalization,
            color=color,
        )
        legendHandles.append(handle)
    if wavelength_range is not None:
        ax[0].set_xlim(wavelength_range[0], wavelength_range[1])
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].set_ylabel("Intensity")
    ax[0].grid(linestyle=(0, (5, 10)))  # offset, dash length, space length
    if legendItemList is not None:
        ax[0].legend(legendHandles, legendItemList)

    for data, column, normalization, color, i in zip(
        dataList, columnList, normalizationList, colors, range(len(dataList))
    ):
        (handle,) = ax[1].semilogy(
            data[:, 7 * column + 1],
            data[:, 7 * column + 2] * normalization + additive_constant_log_intensity,
            color=color,
        )
        legendHandles.append(handle)
    if wavelength_range is not None:
        ax[1].set_xlim(wavelength_range[0], wavelength_range[1])
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_ylabel("Intensity")
    ax[1].grid(linestyle=(0, (5, 10)))
    if legendItemList is not None:
        ax[0].legend(legendHandles, legendItemList)

    fig.savefig(pdfFilename)
    return fig
