"""
This module will contain functions related to the processing of spectra.
"""

from .spectrum import (
    wavelength_to_frequency,
    frequency_to_wavelength,
    transform_limited_pulse_from_spectrometer,
)
from ..personal.marco import (
    load_calibration_data,
    read_spectrometer_excel,
    read_spectrum_ocean_optics,
)
from ..personal.marco import calibrate as calibrate_reso

__all__ = [
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "transform_limited_pulse_from_spectrometer",
    "load_calibration_data",
    "read_spectrometer_excel",
    "read_spectrum_ocean_optics",
    "calibrate_reso",
]
