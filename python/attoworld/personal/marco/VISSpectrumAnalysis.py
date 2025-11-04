import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def box_smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def eliminate_outliers(y, threshold: float = 3, window_points: int = 20):
    """Eliminates outliers in the data by replacing them with the mean of the surrounding values.

    Args:
        y (np.ndarray): Input data array.
        threshold (float): Threshold in units of sigma for outlier detection. Default is 3.
        window_points (int): Number of points to consider for the mean and sigma calculation. Default is 20.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input of eliminate_outliers must be a numpy array.")
    n_outliers = 0
    for i in range(len(y)):
        window_lower = int(max(0, i - window_points / 2))
        window_upper = int(min(len(y), i + window_points / 2))
        mean = np.mean(y[window_lower:window_upper])
        std = np.std(y[window_lower:window_upper])
        if abs(y[i] - mean) > threshold * std:
            y[i] = np.mean(np.concatenate((y[window_lower:i], y[i + 1 : window_upper])))
            n_outliers += 1

    print(
        n_outliers, " outliers detected and replaced with mean of surrounding values."
    )


def read_spectrum_maya(
    filename,
    remove_offsets_individually=False,
    nm_smearing=1.0,
    eliminate_outliers_spectrum=False,
):
    """Reads a spectrum file from the Maya spectrometer and returns the wavelength and spectrum arrays.

    The file is expected to have one column of wavelength data and arbitrarily many columns of equivalent spectra [to be averaged].
    The function also applies a Gaussian filter to the spectrum data to smooth it.
    The function can optionally eliminate outliers in the spectrum data and remove offsets from each column individually.

    Args:
        filename
        remove_offsets_individually (bool): If True, compute and remove offsets from each column individually
        nm_smearing (float): sigma in nm of the gaussian filter applied to smooth the spectrum.
        eliminate_outliers_spectrum (bool): = If True, eliminate single pixel outliers in the spectral data and replace them with average of surrounding values.
            default is False.

    """
    data = pd.read_table(filename, sep=" ", keep_default_na=True, skiprows=0)
    spectrum = []
    offsets = [0.0] * len(data.columns)
    fig, ax = plt.subplots()
    wvl = np.array(data.iloc[:, 0])
    final_nm_smearing = nm_smearing
    if eliminate_outliers_spectrum:
        nm_smearing = 0
    for i in range(len(data.columns) - 1):
        if nm_smearing > 0:
            spectrum_i = gaussian_filter1d(
                np.array(data.iloc[:, i + 1]), sigma=nm_smearing / (wvl[1] - wvl[0])
            )
        else:
            spectrum_i = np.array(data.iloc[:, i + 1])
        if eliminate_outliers_spectrum:
            eliminate_outliers(spectrum_i, threshold=3, window_points=20)
            spectrum_i = gaussian_filter1d(
                spectrum_i, sigma=final_nm_smearing / (wvl[1] - wvl[0])
            )
        offsets[i] = spectrum_i.min()
        if remove_offsets_individually:
            spectrum.append(spectrum_i - offsets[i])
            ax.plot(wvl, spectrum_i - offsets[i], label=f"Channel {i + 1}")
        else:
            spectrum.append(spectrum_i)
            ax.plot(wvl, spectrum_i, label=f"Channel {i + 1}")

    spectrum = np.mean(np.array(spectrum), axis=0)
    ax.plot(wvl, spectrum, label="Mean spectrum")

    return wvl, np.maximum(np.zeros(len(spectrum)), spectrum)


def read_spectrum_ocean_optics(filename):
    """Reads a spectrum file from the Ocean Optics spectrometer and returns the wavelength and spectrum arrays.
    The file is expected to have one column of wavelength data and one column of spectrum data.
    wavelengths are in nm.
    14 lines of header are skipped.
    """
    data = pd.read_table(filename, sep="\t", keep_default_na=True, skiprows=14)
    wvl = np.array(data.iloc[:, 0])
    spectrum = np.array(data.iloc[:, 1])
    return wvl, spectrum


def asymmetric_tukey_f(
    x: float, edge1: float, edge2: float, edge1_width: float, edge2_width: float
):
    if edge1_width < 0 or edge2_width < 0:
        edge1_width = 0
        edge2_width = 0
        print("Warning: negative tukey edge width; rectangular window computed")
    if abs(edge1_width) + abs(edge2_width) > 2 * abs(edge2 - edge1):
        raise Exception("Error: tukey edge width larger than admissible")
    if edge2 < edge1:
        e1 = edge2
        e2 = edge1
        w1 = edge2_width
        w2 = edge1_width
    elif edge1 < edge2:
        e1 = edge1
        e2 = edge2
        w1 = edge1_width
        w2 = edge2_width
    xmin = e1 - w1 / 2
    xmax = e2 + w2 / 2
    if x < xmin or x > xmax:
        y = 0
    elif xmin <= x and x < xmin + w1:
        y = (1 - np.cos((x - xmin) * np.pi / w1)) / 2
    elif xmin + w1 <= x and x <= xmax - w2:
        y = 1
    elif xmax - w2 < x and x <= xmax:
        y = (1 - np.cos((x - xmax) * np.pi / w2)) / 2
    else:
        print("x, xmax, xmin, w1, w2: ", x, xmax, xmin, w1, w2)
        raise ValueError(
            "in tukey_f, x could not be assigned correctly to the sub intervals, might it (or the other parameters) be NaN?"
        )
    return y


def asymmetric_tukey_window(
    x, edge1: float, edge2: float, edge1_width: float, edge2_width: float
):
    if isinstance(x, np.ndarray):
        y = []
        for xi in x:
            y.append(asymmetric_tukey_f(xi, edge1, edge2, edge1_width, edge2_width))
        y = np.array(y)
    elif isinstance(x, list):
        y = []
        for xi in x:
            y.append(asymmetric_tukey_f(xi, edge1, edge2, edge1_width, edge2_width))
    elif isinstance(x, float):
        y = asymmetric_tukey_f(x, edge1, edge2, edge1_width, edge2_width)
    else:
        raise TypeError(
            "in function tukey window x is neither np array nor list nor float"
        )
    return y


class SpectrumHandler:
    """Class to perform standard operations on spectra (calibration, plotting, division, multiplication, etc.)."""

    def __init__(
        self,
        filename: str = None,
        wavelengths=None,
        spectrum=None,
        remove_offsets_individually: bool = False,
        nm_smearing=1.0,
        eliminate_outliers_spectrum=False,
        filetype="MayaScarab",
    ):
        """Provide either a filename or the wavelengths and spectrum arrays to the constructor.

        Args:
            filename (str): Path to the file containing the spectrum data.
            wavelengths and spectrum (np.ndarray): = Wavelengths and spectrum data arrays. (alternative to filename)
            remove_offsets_individually (bool): = If True, compute and remove offsets from each column individually
                (filename is expected to have one column of wavelength data and arbitrarily many columns of equivalent spectra [to be averaged]).
            nm_smearing (float): Smearing in nm to be applied to the spectrum.
            eliminate_outliers_spectrum (bool): If True, eliminate single pixel outliers in the spectral data and replace them with average of surrounding values.
            filetype (str): Type of the file to be read. Options are 'MayaScarab' (default) or 'OceanOptics'.

        """
        self.nm_smearing = nm_smearing
        self.wvl, self.spectrum = np.linspace(200, 1000, 800), np.zeros(800)
        self.calibration_lamp_wvl, self.calibration_lamp_spectrum = None, None
        self.calibration_factor = None

        if filetype == "MayaScarab" and filename is not None:
            self.wvl, self.spectrum = read_spectrum_maya(
                filename,
                remove_offsets_individually,
                nm_smearing,
                eliminate_outliers_spectrum,
            )
        elif filetype == "OceanOptics" and filename is not None:
            self.wvl, self.spectrum = read_spectrum_ocean_optics(filename)
        elif wavelengths is not None and spectrum is not None:
            self.wvl = wavelengths
            self.spectrum = spectrum
        self.check_wvl_ascending()

    def get_spectrum(self):
        return self.wvl, self.spectrum

    def get_calibration_factor(self):
        if self.calibration_factor is None:
            print(
                "Warning: Calibration factor not available. Computing it from given spectra..."
            )
            self.compute_calibration_factor_spectrometer()
        return self.wvl, self.calibration_factor

    def calibrate(self):
        """Applies the calibration factor to the spectrum and sets the calibration factor to None."""
        if self.calibration_factor is None:
            print(
                "Warning: Calibration factor not available. Try load_calibration_factor_from_file(). Returning original spectrum."
            )
        else:
            self.spectrum = self.spectrum * self.calibration_factor
            self.calibration_factor = None
            print(
                "Calibration factor applied to the spectrum. Calibration factor is now set to None."
            )

    def save_to_file(self, filename):
        if self.wvl is None or self.spectrum is None:
            raise ValueError("Spectrum not loaded.")
        data = pd.DataFrame({"Wavelength": self.wvl, "Spectrum": self.spectrum})
        data.to_csv(filename, index=False, sep="\t")

    def save_calibration_factor_to_file(self, filename):
        """Saves the calibration factor to a file.

        The file can be either a .txt, .csv or .npz file.
        The file will contain the wavelength, calibration factor, calibration lamp true spectrum and measured lamp spectrum.
        """
        if self.calibration_factor is None:
            raise ValueError("Calibration factor not computed.")
        if ".txt" in filename or ".csv" in filename:
            data = pd.DataFrame(
                {
                    "Wavelength": self.wvl,
                    "Calibration Factor": self.calibration_factor,
                    "Lamp Spectrum": self.calibration_lamp_spectrum,
                    "Measured Spectrum": self.spectrum,
                }
            )
            data.to_csv(filename, index=False, sep="\t")
        elif ".npz" in filename:
            np.savez(
                filename,
                wavelength=self.wvl,
                corr_factor_smoothed=self.calibration_factor,
                lamp_ref=self.calibration_lamp_spectrum,
                lamp_measured=self.spectrum,
            )
        else:
            raise ValueError("File type not supported. Use .txt, .csv or .npz.")

    def load_calibration_factor_from_file(self, filename):
        data = pd.read_table(filename, sep="\t", keep_default_na=True, skiprows=0)
        self.calibration_factor = np.interp(
            self.wvl, np.array(data.iloc[:, 0]), np.array(data.iloc[:, 1])
        )

    def check_wvl_ascending(self):
        if self.wvl is not None and np.any(np.diff(self.wvl)) < 0:
            if np.all(np.diff(self.wvl) < 0):
                self.wvl = np.flip(self.wvl)
                self.spectrum = np.flip(self.spectrum)
            else:
                raise ValueError(
                    "Wavelength array must be either in ascending or descending order."
                )
        # same for calib spectrum
        if (
            self.calibration_lamp_wvl is not None
            and np.any(np.diff(self.calibration_lamp_wvl)) < 0
        ):
            if np.all(np.diff(self.calibration_lamp_wvl) < 0):
                self.calibration_lamp_wvl = np.flip(self.calibration_lamp_wvl)
                self.calibration_lamp_spectrum = np.flip(self.calibration_lamp_spectrum)
            else:
                raise ValueError(
                    "Wavelength array must be either in ascending or descending order."
                )
        return True

    def load_calibration_lamp_data(
        self, filename="./calibration_data/7315273LS-Deuterium-Halogen_CC-VISNIR.lmp"
    ):
        Data = pd.read_table(filename)
        self.calibration_lamp_wvl, self.calibration_lamp_spectrum = (
            np.array(Data.iloc[:, 0]),
            np.array(Data.iloc[:, 1]),
        )
        self.calibration_lamp_spectrum = gaussian_filter1d(
            self.calibration_lamp_spectrum,
            self.nm_smearing
            / (self.calibration_lamp_wvl[1] - self.calibration_lamp_wvl[0]),
        )
        self.check_wvl_ascending()

    def plot_calibration_data(
        self,
        low_lim=None,
        up_lim=None,
        low_lim_y=None,
        up_lim_y=None,
        wavelength_ROI: list = [420, 800],
    ):
        """Plot the calibration data of the lamp, the measured lamp spectrum, and the calibration factor.

        The method assumes that the true calibration lamp data and the measured spectrum of the calibration lamp are loaded in the object
        The normalization of the three curves is arbitrary. By default the region between 420 and 800 nm is used for normalization.

        Args:
            low_lim, up_lim(float): Wavelength limits for the x-axis.
            low_lim_y, up_lim_y (float): Intensity limits for the y-axis.
            wavelength_ROI (list): [wvl1, wvl2] = Wavelength range for the normalization of the calibration factor (for display purposes).
                Default is [420, 800] nm.

        """
        if low_lim is None or up_lim is None:
            low_lim = np.min(self.wvl)
            up_lim = np.max(self.wvl)
        if low_lim_y is None or up_lim_y is None:
            low_lim_y = 0
            up_lim_y = 3
        fig, ax = plt.subplots()
        ax.plot(
            self.wvl,
            self.spectrum / np.max(self.spectrum),
            label="spectrometer reading",
        )
        multipl_factor_lamp_plot = (
            self.spectrum[
                (self.wvl > wavelength_ROI[0]) & (self.wvl < wavelength_ROI[1])
            ].mean()
            / self.calibration_lamp_spectrum[
                (self.calibration_lamp_wvl > wavelength_ROI[0])
                & (self.calibration_lamp_wvl < wavelength_ROI[1])
            ].mean()
        )

        ax.plot(
            self.calibration_lamp_wvl,
            self.calibration_lamp_spectrum
            / np.max(self.spectrum)
            * multipl_factor_lamp_plot,
            label="actual lamp spectrum",
        )
        if self.calibration_factor is not None:
            ax.plot(self.wvl, self.calibration_factor, label="calibration factor")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        ax.legend(loc="upper right")
        ax.set_xlim(low_lim, up_lim)
        ax.set_ylim(low_lim_y, up_lim_y)
        return fig

    def calibrate_wavelength_axis(self, intercept: float, slope: float):
        """Calibrate the wavelength axis of the spectrum using the given coefficient of a linear fit.

        Form of the calibration: true_wavelength = intercept + slope * measured_wavelength
        The function modifies the wavelength axis in place.

        Args:
            intercept: float,
            slope: float

        """
        if self.wvl is None or self.spectrum is None:
            raise ValueError("Spectrum not loaded.")
        self.wvl = intercept + slope * self.wvl
        self.check_wvl_ascending()

    def tukey_filter(
        self, edge1: float, edge2: float, edge1_width: float, edge2_width: float
    ):
        """Applies a Tukey window to the spectrum (e.g. to remove noise at the edges).

        Tukey window is defined as:
         * cosine-shaped window between edge1-edge1_width/2 and edge1+edge1_width/2
         * 1 between edge1+edge1_width/2 and edge2-edge2_width/2
         * cosine-shaped window between edge2-edge2_width/2 and edge2+edge2_width/2
         * 0 everywhere else
        In this case, tukey window is computed in the wavelength domain.

        Args:
            edge1, edge2 (float): Wavelength limits for the Tukey window (edge2-edge1 = FWHM, in nm)
            edge1_width, edge2_width (float): Width of the cosine-shaped edge of the tukey window (in nm).

        """
        window = asymmetric_tukey_window(
            self.wvl, edge1, edge2, edge1_width, edge2_width
        )
        self.spectrum = self.spectrum * window
        if self.calibration_factor is not None:
            self.calibration_factor = self.calibration_factor * window
        if (
            self.calibration_lamp_wvl is not None
            and self.calibration_lamp_spectrum is not None
        ):
            window = asymmetric_tukey_window(
                self.calibration_lamp_wvl, edge1, edge2, edge1_width, edge2_width
            )
            self.calibration_lamp_spectrum = self.calibration_lamp_spectrum * window

    def divide_by(self, spectrumObject, nm_smearing=0.0):
        """Normalizes the current spectrum with another spectrum, and stores the result in the present object.

        Args:
            spectrumObject (either 'c'/'calib'/'calibration'/'lamp' or SpectrumHandler object):
                the current spectrum is divided at each wavelength by the loaded calibration lamp data or by the spectrum passed as SpectrumHandler object.
            nm_smearing (float): sigma in nm for the gaussian smoothing

        """
        if (
            spectrumObject == "c"
            or spectrumObject == "calib"
            or spectrumObject == "calibration"
            or spectrumObject == "lamp"
            or spectrumObject == "calibration data"
        ):
            if (
                self.calibration_lamp_wvl is None
                or self.calibration_lamp_spectrum is None
            ):
                raise ValueError("Calibration lamp data not loaded.")
            wvl_divisor = self.calibration_lamp_wvl
            spectrum_divisor = self.calibration_lamp_spectrum
        elif not isinstance(spectrumObject, SpectrumHandler):
            raise TypeError("Input of divide_by must be a SpectrumHandler object.")
        else:
            wvl_divisor, spectrum_divisor = spectrumObject.get_spectrum()

        if nm_smearing > 0:
            spectrum_divisor = gaussian_filter1d(
                spectrum_divisor, sigma=nm_smearing / (wvl_divisor[1] - wvl_divisor[0])
            )
            self.spectrum = gaussian_filter1d(
                self.spectrum, sigma=nm_smearing / (self.wvl[1] - self.wvl[0])
            )
        commonWvl = self.wvl[
            (self.wvl > np.min(wvl_divisor)) & (self.wvl < np.max(wvl_divisor))
        ]
        interpolated_spd = np.interp(commonWvl, wvl_divisor, spectrum_divisor)
        interpolated_original = np.interp(commonWvl, self.wvl, self.spectrum)
        self.wvl = commonWvl
        self.spectrum = interpolated_original / interpolated_spd

    def add(self, spectrumObject):
        """Adds two spectra and stores the result in the present object.

        Args:
            spectrumObject (SpectrumHandler): spectrum to be added

        """
        if not isinstance(spectrumObject, SpectrumHandler):
            raise TypeError("Input of add must be a SpectrumHandler object.")
        wvl_add, spectrum_add = spectrumObject.get_spectrum()
        commonWvl = self.wvl[
            (self.wvl > np.min(wvl_add)) & (self.wvl < np.max(wvl_add))
        ]
        interpolated_spd = np.interp(commonWvl, wvl_add, spectrum_add)
        interpolated_original = np.interp(commonWvl, self.wvl, self.spectrum)
        self.wvl = commonWvl
        self.spectrum = interpolated_original + interpolated_spd

    def multiply(self, spectrumObject):
        """Multiplies the current spectrum by a second one at each wavelength.

        Args:
            spectrumObject (SpectrumHandler)

        """
        if not isinstance(spectrumObject, SpectrumHandler):
            raise TypeError("Input of multiply must be a SpectrumHandler object.")
        wvl_mult, spectrum_mult = spectrumObject.get_spectrum()
        commonWvl = self.wvl[
            (self.wvl > np.min(wvl_mult)) & (self.wvl < np.max(wvl_mult))
        ]
        interpolated_spd = np.interp(commonWvl, wvl_mult, spectrum_mult)
        interpolated_original = np.interp(commonWvl, self.wvl, self.spectrum)
        self.wvl = commonWvl
        self.spectrum = interpolated_original * interpolated_spd

    def multiply_scalar(self, scalar):
        """Multiplies the current spectrum by a constant.

        Args:
            scalar (float)

        """
        self.spectrum = self.spectrum * scalar

    def add_scalar(self, scalar):
        """Adds a constant offset.

        Args:
            scalar (float)

        """
        self.spectrum = self.spectrum + scalar

    def compute_calibration_factor_spectrometer(
        self,
        transmission_additional_optics=None,
        smoothing="poly",
        extend_calibration: bool = False,
        wavelength_ROI: list = [420, 800],
    ):
        """Computes the calibration factor.

        It assumes that in the SpectrumHandler object:
        * the measured spectrum of the calibration lamp is loaded as main spectrum
        * the true tabulated spectrum of the calibration lamp is loaded as self.calibration_lamp_spectrum [via load_calibration_lamp_data()]

        Args:
           transmission_additional_optics: list of SpectrumHandler objects = list of transmission spectra of additional optics to be taken into account when computing the calibration factor.
               The transmission ['reflection' for mirrors] spectra of additional optics will be included (multiplied) in the calibration factor
               Additional optics are assumed to be in the beam path of any standard measurement (e.g. integrating sphere) BUT NOT PRESENT during the calibration lamp measurement.
           smoothing: Type of smoothing to be applied to the calibration factor. Options are 'poly' (default) None or an integer number. If 'poly', a polynomial fit is applied to the calibration factor.
               If None, no smoothing is applied. If an integer number, a box filter of that size is applied to the calibration factor.
           extend_calibration (bool): If True, the calibration factor is extended to the full wavelength range of the spectrometer (the first and last values of the calibration curve are used for the extension).
               This is a quick and dirty solution to deal with the fact that the calibration lamp spectrum is not available for the full wavelength range of the spectrometer.
           wavelength_ROI (list): [wvl1, wvl2] Wavelength range for the normalization of the calibration factor (for display purposes).
               Default is [420, 800] nm.

        """
        if self.calibration_lamp_wvl is None or self.calibration_lamp_spectrum is None:
            raise ValueError("Calibration lamp data not loaded.")
        wvl_stored, spectrum_stored = self.wvl, self.spectrum
        self.divide_by("calibration data")

        eliminate_outliers(self.spectrum, threshold=3, window_points=20)

        if transmission_additional_optics is not None:
            for sao in transmission_additional_optics:
                if not isinstance(sao, SpectrumHandler):
                    raise TypeError(
                        "Input of spectra_additional_optics must be a list of SpectrumHandler objects."
                    )
                wvl_sao, spectrum_sao = sao.get_spectrum()
                if np.any(np.diff(wvl_sao)) < 0:
                    raise ValueError(
                        "Wavelength array of additional optics spectrum must be sorted in ascending order."
                    )
                wvl_sao = np.concatenate(
                    (
                        np.array([0]),
                        wvl_sao,
                        np.array([max(np.max(self.wvl), np.max(wvl_sao))]),
                    )
                )
                spectrum_sao = np.concatenate(
                    (
                        np.array([spectrum_sao[0]]),
                        spectrum_sao,
                        np.array([spectrum_sao[-1]]),
                    )
                )
                self.spectrum = self.spectrum * np.interp(
                    self.wvl, wvl_sao, spectrum_sao
                )

        self.calibration_factor = self.spectrum
        if smoothing is None:
            pass
        elif smoothing == "poly":
            smoothed_cf = np.poly1d(np.polyfit(self.wvl, self.calibration_factor, 50))
            self.calibration_factor = smoothed_cf(self.wvl)
        else:
            if not isinstance(smoothing, int):
                raise TypeError(
                    "Input of smoothing must be 'poly', None or an integer."
                )
            if smoothing < 1:
                raise ValueError("Input of smoothing must be a positive integer.")
            self.calibration_factor = box_smooth(self.calibration_factor, smoothing)

        for i in range(len(self.wvl)):
            if self.calibration_factor[i] < 0.01:
                self.calibration_factor[i] = 0.01
        self.calibration_factor = 1 / self.calibration_factor
        self.calibration_factor = self.calibration_factor / np.mean(
            self.calibration_factor[
                (self.wvl > wavelength_ROI[0]) & (self.wvl < wavelength_ROI[1])
            ]
        )

        if extend_calibration:
            # Extend the calibration factor to the full wavelength range of the spectrometer (pick the first and last values of the calibration curve for the extension)
            self.wvl = np.concatenate(
                (
                    np.linspace(0, self.wvl[0], 100),
                    self.wvl,
                    np.linspace(self.wvl[-1], self.wvl[-1], 100),
                )
            )
            self.calibration_factor = np.concatenate(
                (
                    np.ones(100) * self.calibration_factor[0],
                    self.calibration_factor,
                    np.ones(100) * self.calibration_factor[-1],
                )
            )
            self.calibration_factor = np.interp(
                wvl_stored, self.wvl, self.calibration_factor
            )
            self.wvl = wvl_stored
            self.spectrum = spectrum_stored
        else:
            self.spectrum = spectrum_stored[
                (wvl_stored > np.min(self.calibration_lamp_wvl))
                & (wvl_stored < np.max(self.calibration_lamp_wvl))
            ]

    def plot_spectrum(self, low_lim=None, up_lim=None):
        fig, ax = plt.subplots()
        if low_lim is not None and up_lim is not None:
            ax.plot(
                self.wvl[(self.wvl > low_lim) & (self.wvl < up_lim)],
                self.spectrum[(self.wvl > low_lim) & (self.wvl < up_lim)],
            )
        else:
            ax.plot(self.wvl, self.spectrum)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        return fig


class MultiSpectrumHandler:
    """Stores and plots multiple spectra."""

    def __init__(
        self,
        filenameList: list = None,
        wavelengthList: list = None,
        spectrumList: list = None,
        spectrumHandlerList: list = None,
        remove_offsets_individually: bool = False,
        nm_smearing=1.0,
        eliminate_outliers_spectrum=False,
        filetype="MayaScarab",
    ):
        """Constructor for the MultiSpectrumHandler class.

        Args:
            filenameList: list = List of filenames to be read.
            wavelengthList: list = List of wavelength arrays. (alternative to filename)
            spectrumList: list = List of spectrum arrays. (alternative to filename)
            remove_offsets_individually: bool = If True, compute and remove offsets from each column individually
                (filename is expected to have one column of wavelength data and arbitrarily many columns of equivalent spectra [to be averaged]).
            nm_smearing: float = Smearing in nm to be applied to the spectrum.
            eliminate_outliers_spectrum: bool = If True, eliminate single pixel outliers in the spectral data and replace them with average of surrounding values.
            filetype: str = Type of the file to be read. Options are 'MayaScarab' (default) or 'OceanOptics'.

        """
        self.spectrumHandlers = []
        if filenameList is not None:
            for filename in filenameList:
                self.spectrumHandlers.append(
                    SpectrumHandler(
                        filename=filename,
                        remove_offsets_individually=remove_offsets_individually,
                        nm_smearing=nm_smearing,
                        eliminate_outliers_spectrum=eliminate_outliers_spectrum,
                        filetype=filetype,
                    )
                )
        elif wavelengthList is not None and spectrumList is not None:
            for i in range(len(wavelengthList)):
                self.spectrumHandlers.append(
                    SpectrumHandler(
                        wavelengths=wavelengthList[i],
                        spectrum=spectrumList[i],
                        remove_offsets_individually=remove_offsets_individually,
                        nm_smearing=nm_smearing,
                        eliminate_outliers_spectrum=eliminate_outliers_spectrum,
                        filetype=filetype,
                    )
                )
        elif spectrumHandlerList is not None:
            for spectrumHandler in spectrumHandlerList:
                if not isinstance(spectrumHandler, SpectrumHandler):
                    raise TypeError(
                        "Input of spectrumHandlerList must be a list of SpectrumHandler objects."
                    )
                self.spectrumHandlers.append(spectrumHandler)
        else:
            raise ValueError(
                "Either filenameList or wavelengthList and spectrumList or spectrumHandlerList must be provided. filetype, if provided, can only be 'MayaScarab' or 'OceanOptics'."
            )

    def plot(self, low_lim=None, up_lim=None):
        fig, ax = plt.subplots()
        for i, spectrumHandler in enumerate(self.spectrumHandlers):
            wvl, spectrum = spectrumHandler.get_spectrum()
            if low_lim is not None and up_lim is not None:
                ax.plot(
                    wvl[(wvl > low_lim) & (wvl < up_lim)],
                    spectrum[(wvl > low_lim) & (wvl < up_lim)],
                    label=f"Spectrum {i + 1}",
                )
            else:
                ax.plot(wvl, spectrum, label=f"Spectrum {i + 1}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Arb. unit)")
        ax.legend(loc="upper right")
        return fig
