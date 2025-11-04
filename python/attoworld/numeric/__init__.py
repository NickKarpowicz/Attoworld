"""A collection of numerical tools."""

from ..attoworld_rs import (
    derivative,
    derivative_periodic,
    find_first_intercept,
    find_last_intercept,
    find_maximum_location,
    fornberg_stencil,
    fwhm,
    interpolate,
)
from .atomic_units import AtomicUnits
from .numeric import block_binning_1d, block_binning_2d

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
