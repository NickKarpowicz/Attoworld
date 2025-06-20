"""
This module will contain numerical tools.
"""

from .atomic_units import AtomicUnits
from .numeric import block_binning_1d, block_binning_2d
from ..attoworld_rs import (
    fornberg_stencil,
    interpolate,
    fwhm,
    find_first_intercept,
    find_last_intercept,
    find_maximum_location,
    derivative,
    derivative_periodic,
)

__all__ = [
    "fornberg_stencil",
    "find_maximum_location",
    "fwhm",
    "find_first_intercept",
    "find_last_intercept",
    "derivative",
    "derivative_periodic",
    "interpolate",
    "AtomicUnits",
    "block_binning_1d",
    "block_binning_2d",
]
