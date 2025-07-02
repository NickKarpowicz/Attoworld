from enum import Enum
import importlib.resources


def get_calibration_path():
    with importlib.resources.path(__name__) as data_path:
        return data_path


class CalibrationData(Enum):
    MPQ_ATTO_RESO_MARCO = "MPQ_Atto_Reso_Spectrometer_Marco.npz"


class CalibrationLampReferences(Enum):
    MPQ_ATTO_DEUTERIUM_HALOGEN = "7315273LS-Deuterium-Halogen_CC-VISNIR.lmp"
    MPQ_ATTO_HALOGEN = "7315273LS-Halogen_CC-VISNIR.lmp"
