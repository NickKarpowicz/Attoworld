"""
This module will contain functions related to the processing of spectra.
"""
from .spectrum import wavelength_to_frequency, transform_limited_pulse_from_spectrometer
from .UVSpectrumAnalysis import load_calibration_data, smooth, tukey_f, tukey_window, read_csd_file, plot_spectra, read_spectrometer_excel, calibrate, plot_spectra_UVsp
from .VISSpectrumAnalysis import eliminate_outliers, read_spectrum_maya, read_spectrum_ocean_optics, asymmetric_tukey_f, asymmetric_tukey_window, SpectrumHandler, MultiSpectrumHandler

__all__ = [
    "wavelength_to_frequency"
    "transform_limited_pulse_from_spectrometer"
    "load_calibration_data",
    "smooth",
    "tukey_f",
    "tukey_window",
    "read_csd_file",
    "plot_spectra",
    "read_spectrometer_excel",
    "calibrate",
    "plot_spectra_UVsp",
    "eliminate_outliers",
    "read_spectrum_maya",
    "read_spectrum_ocean_optics",
    "asymmetric_tukey_f",
    "asymmetric_tukey_window",
    "SpectrumHandler",
    "MultiSpectrumHandler"
]
