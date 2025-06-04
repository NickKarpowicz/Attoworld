import numpy as np
from ..attoworld_rs import fornberg_stencil, interpolate_sorted_1d


def interpolate(x_out:np.ndarray, x_in: np.ndarray, y_in:np.ndarray, neighbors: int = 3, extrapolate: bool = False, derivative_order: int = 0, input_is_sorted: bool = True) -> np.ndarray:
    """
    Use a Fornberg stencil containing a specified number of neighboring points to perform interpolation.

    Args:
        x_out (np.ndarray): array of output x values, the array onto which y_in will be interpolated
        x_in (np.ndarray): array of input x values
        y_in (np.ndarray): array of input y values
        neighbors (int): number of nearest neighbors to include in the interpolation
        extrapolate (bool): unless set to true, values outside of the range of x_in will be zero
        derivative_order(int): order of derivative to take. 0 (default) is plain interpolation, 1 takes first derivative, and so on.
        input_is_sorted (bool): if set to false, data will be sorted before extrapolation
    Returns:
        np.ndarray: the interpolated y_out
    """
    if input_is_sorted:
        return interpolate_sorted_1d(
            x_out,
            x_in,
            y_in,
            np.searchsorted(x_in, x_out, side='left'),
            neighbors,
            extrapolate,
            derivative_order)
    sort_order = np.argsort(x_in)
    x_in_sorted = x_in[sort_order]
    y_in_sorted = y_in[sort_order]
    return interpolate_sorted_1d(
        x_out,
        x_in_sorted,
        y_in_sorted,
        np.searchsorted(x_in_sorted, x_out, side='left'),
        neighbors,
        extrapolate,
        derivative_order)
