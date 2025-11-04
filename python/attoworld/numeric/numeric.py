"""Numerical tools implemented in python (most are in Rust here now)."""

import numpy as np


def block_binning_1d(data: np.ndarray, bin_size: int, method: str) -> np.ndarray:
    """Perform binning on an array, reducing its size by averaging (bin) blocks.

    Args:
        data (np.ndarray): the 1d array to bin
        bin_size (int): the size of bins
        method (str): the method for averaging: ```mean``` (default) or ```median```

    Returns:
        np.ndarray: the binned data

    """
    new_shape = (data.shape[0] // bin_size, bin_size)
    truncate = bin_size * (data.shape[0] // bin_size)
    reshaped_data = data[0:truncate].reshape(new_shape)
    match method:
        case "median":
            return np.median(reshaped_data, axis=1)
        case "mean" | _:
            return np.mean(reshaped_data, axis=1)


def block_binning_2d(
    data: np.ndarray, x_bin: int = 2, y_bin: int = 2, method: str = "mean"
) -> np.ndarray:
    """Perform binning on an array, reducing its size by averaging (x_bin x y_bin) blocks.

    Args:
        data (np.ndarray): the 2d array to bin
        x_bin (int): the size of bins in the x-direction
        y_bin (int): the size of bins in the y-direction
        method (str): the method for averaging: ```mean``` (default) or ```median```

    Returns:
        np.ndarray: the binned data

    """
    new_shape = (data.shape[0] // y_bin, y_bin, data.shape[1] // x_bin, x_bin)
    truncate_x = x_bin * (data.shape[1] // x_bin)
    truncate_y = y_bin * (data.shape[0] // y_bin)
    reshaped_data = data[0:truncate_y, 0:truncate_x].reshape(new_shape)

    match method:
        case "median":
            return np.median(reshaped_data, axis=(1, 3))
        case "mean" | _:
            return np.mean(reshaped_data, axis=(1, 3))
