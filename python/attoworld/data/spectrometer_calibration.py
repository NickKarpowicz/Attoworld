"""Define the base class for handling spectrometer calibrations."""

from dataclasses import dataclass

import numpy as np

from .. import spectrum
from .decorators import yaml_io


@yaml_io
@dataclass
class SpectrometerCalibration:
    """Set of data describing a spectrometer calibration.

    Attributes:
        intensity_factors (np.ndarray): the intensity correction factors (weights)
        corrected_wavelengths (np.ndarray): the corrected wavelength array of the spectrometer
        corrected_frequencies (np.ndarray): the corrected frequency array of the spectrometer (c/wavelengths)

    """

    intensity_factors: np.ndarray
    original_wavelengths: np.ndarray
    corrected_wavelengths: np.ndarray
    corrected_frequencies: np.ndarray

    def save_npz(self, filepath):
        """Save to an npz file.

        Args:
            filepath: path to save to

        """
        np.savez(
            filepath,
            intensity_factors=self.intensity_factors,
            original_wavelengths=self.original_wavelengths,
            corrected_wavelengths=self.corrected_wavelengths,
            corrected_frequencies=self.corrected_frequencies,
        )

    @staticmethod
    def from_npz(filepath):
        """Make an instance of the class from an npz file.

        Args:
            filepath: path to the file

        Returns:
            SpectrometerCalibration: the calibration in the file

        """
        npzfile = np.load(filepath)
        return SpectrometerCalibration(
            intensity_factors=npzfile["intensity_factors"],
            original_wavelengths=npzfile["original_wavelengths"],
            corrected_wavelengths=npzfile["corrected_wavelengths"],
            corrected_frequencies=npzfile["corrected_frequencies"],
        )

    @staticmethod
    def from_named(spectrometer: spectrum.CalibrationData):
        """Loads a calibration saved in the database.

        Args:
            spectrometer (spectrum.CalibrationData): Value from the CalibrationData enum attoworld.spectrum.CalibrationData

        Returns:
            SpectrometerCalibration: the calibration associated with the enum value

        """
        return SpectrometerCalibration.from_npz(
            spectrum.get_calibration_path() / spectrometer.value
        )
