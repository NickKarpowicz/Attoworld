from enum import Enum
import importlib.resources


def get_calibration_path():
    with importlib.resources.path(__name__) as data_path:
        return data_path


class CalibrationData(Enum):
    MPQ_ATTO_RESO_MARCO = "MPQ_Atto_Reso_Spectrometer_Marco.npz"
