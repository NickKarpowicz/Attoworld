from enum import Enum
import importlib.resources


def get_calibration_path():
    with importlib.resources.path(__name__) as data_path:
        return data_path


class CalibrationData(Enum):
    mpq_atto_reso_marco = "MPQ_Atto_Reso_Spectrometer_Marco.npz"


class CalibrationLampReferences(Enum):
    mpq_atto_deuterium_halogen = "7315273LS-Deuterium-Halogen_CC-VISNIR.lmp"
    mpq_atto_halogen = "7315273LS-Halogen_CC-VISNIR.lmp"
