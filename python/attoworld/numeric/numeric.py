import numpy as np
from ..attoworld_rs import fornberg_stencil

def uniform_derivative(data: np.ndarray, order: int = 1, neighbors: int = 1, boundary: str = 'internal') -> np.ndarray:
    """
    Use a Fornberg stencil to take a derivative of arbitrary order and accuracy, handling the edge
    by using modified stencils that only use internal points.

    Args:
        data (np.ndarray): the data whose derivative should be taken
        order (int): the order of the derivative
        neighbors (int): the number of nearest neighbors to consider.
        boundary (str): How to treat the boundary: 'internal' will use only internal points (default). 'periodic' will assume periodic boundary. 'zero' will assume the data is zero outsize the grid.

    Returns:
        np.ndarray: the derivative
    """

    positions = np.array(range(-neighbors,neighbors+1))
    stencil = fornberg_stencil(order, positions)
    derivative = np.convolve(data, np.flip(stencil), mode='same')

    match boundary:
        case 'zero':
            return derivative
        case 'periodic':
            boundary_array = np.concatenate((data[-2*neighbors::],data[0:(2*neighbors + 1)]))
            for _i in range(2*neighbors + 1):
                derivative[-neighbors + _i] = np.sum(boundary_array[_i:(_i + 2*neighbors + 1)] * stencil)
            return derivative
        case 'internal':
            # increase number of included neighbors to improve accuracy
            neighbors += 1
            positions = np.array(range(-neighbors,neighbors+1))
            def corrected_point_top(index: int):
                boundary_stencil = fornberg_stencil(order, positions + neighbors - index)
                return np.sum(boundary_stencil*data[0:len(positions)])

            def corrected_point_bottom(index: int):
                boundary_stencil = fornberg_stencil(order, positions-neighbors+index)
                return np.sum(boundary_stencil*data[(-len(positions))::])

            for _i in range(neighbors):
                derivative[_i] = corrected_point_top(_i)
                derivative[-1 -_i] = corrected_point_bottom(_i)

    return derivative

def interpolate(x_out:np.ndarray, x_in: np.ndarray, y_in:np.ndarray, neighbors: int = 2, extrapolate: bool = False) -> np.ndarray:
    """
    Use a Fornberg stencil containing a specified number of neighboring points to perform interpolation.

    Contributed by Nick Karpowicz

    Args:
        x_out (np.ndarray): array of output x values, the array onto which y_in will be interpolated
        x_in (np.ndarray): array of input x values
        y_in (np.ndarray): array of input y values
        neighbors (int): number of nearest neighbors to include in the interpolation
        extrapolate (bool): unless set to true, values outside of the range of x_in will be zero

    Returns:
        np.ndarray: the interpolated y_out
    """
    sort_order = np.argsort(x_in)
    x_in_sorted = x_in[sort_order]
    y_in_sorted = y_in[sort_order]
    y_out = np.zeros(x_out.shape)
    locations = np.searchsorted(x_in_sorted, x_out, side='left')

    def interpolate_front_edge(x):
        stencil = fornberg_stencil(
            0,
            x_in_sorted[0:(2*neighbors)],
            x)
        return np.sum(stencil * y_in_sorted[0:(2*neighbors)])

    def interpolate_rear_edge(x):
        stencil = fornberg_stencil(
            0,
            x_in_sorted[(-1 - 2*neighbors)::],
            x)
        return np.sum(stencil * y_in_sorted[(-1 - 2*neighbors)::])

    def interpolate_point(x, location):

        if (((location == 0) and x != x_in_sorted[0]) or (location >= len(x_in_sorted))):
            #points outside the range of x_in (extrapolation)
            if extrapolate:
                if location == 0:
                    return interpolate_front_edge(x)
                else:
                    return interpolate_rear_edge(x)
            else:
                return 0.0

        elif x == x_in_sorted[location]:
            #case if x is exactly in the x_in array
            #if multiple points match, return average
            return np.mean(y_in_sorted[location==x_in_sorted])

        elif location < neighbors:
            #use modified front-edge interp
            return interpolate_front_edge(x)

        elif location >= (len(x_in_sorted) - neighbors):
            #use modified rear-edge interp
            return interpolate_rear_edge(x)
        else:
            #normal interior point
            stencil = fornberg_stencil(
                0,
                x_in_sorted[(location-neighbors):(location + neighbors)],
                x)
            return np.sum(stencil * y_in_sorted[(location-neighbors):(location + neighbors)])

    for _i in range(len(x_out)):
        y_out[_i] = interpolate_point(x_out[_i], locations[_i])

    return y_out
