"""Contains functions related to the processing of spectra."""

from ..personal.marco import (
    load_calibration_data,
    read_spectrometer_excel,
    read_spectrum_ocean_optics,
)
from .calibration_data import (
    CalibrationData,
    CalibrationLampReferences,
    get_calibration_path,
)
from .spectrum import (
    frequency_to_wavelength,
    transform_limited_pulse_from_spectrometer,
    wavelength_to_frequency,
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
