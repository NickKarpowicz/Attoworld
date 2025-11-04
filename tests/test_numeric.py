"""Test the numeric functions."""

import attoworld as aw
import numpy as np


def test_fornberg_stencil():
    """Test that the fornberg stencil is correct."""
    np.testing.assert_allclose(
        aw.numeric.fornberg_stencil(2, np.array([-1.0, 0.0, 1.0])),
        np.array([1.0, -2.0, 1.0]),
    )


def test_fwhm():
    """Check that the fwhm of a Gaussian is as expected."""
    x = np.linspace(-5, 5, 65)
    dx = x[1] - x[0]
    y = np.exp(-(x**2) / 2)
    fwhm = aw.numeric.fwhm(y, dx)
    np.testing.assert_allclose(fwhm, 2.3547421160236)
