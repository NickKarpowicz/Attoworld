"""Organize the calibration data we have available."""

import importlib.resources
from enum import Enum


def get_calibration_path():
    """Return the absolute path to the calibration files in the module."""
    with importlib.resources.path(__name__) as data_path:
        return data_path


class CalibrationData(Enum):
    """Contains a list of the calibration files store in the module."""

    mpq_atto_reso_marco = "MPQ_Atto_Reso_Spectrometer_Marco.npz"
    mpq_atto_maya_1184_1101 = "MPQ_Atto_Maya_184-1101.npz"
    mpq_atto_frog_ocean_hdx_xr = "MPQ_Atto_FROG_Ocean_HDX-XR.npz"


class CalibrationLampReferences(Enum):
    """Contains a list of the calibration lamp references stored in the module."""

    mpq_atto_deuterium_halogen = "7315273LS-Deuterium-Halogen_CC-VISNIR.lmp"
    mpq_atto_halogen = "7315273LS-Halogen_CC-VISNIR.lmp"
