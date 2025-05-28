"""
This module will contain functions for plotting, and for formatting plots, with the goal of
more easily obtaining a consistent style across plots made by different co-authors.
"""

from .plot import *
__all__ = [name for name in globals() if callable(globals()[name]) and not name.startswith("_")]
