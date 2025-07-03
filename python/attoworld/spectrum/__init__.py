"""
This module will contain functions related to the processing of spectra.
"""

from .calibration_data import (
    CalibrationData,
    get_calibration_path,
    CalibrationLampReferences,
)

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

__all__ = [
    "CalibrationData",
    "CalibrationLampReferences",
    "get_calibration_path",
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "transform_limited_pulse_from_spectrometer",
    "load_calibration_data",
    "read_spectrometer_excel",
    "read_spectrum_ocean_optics",
]
