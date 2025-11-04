"""Tools from the Attosecond science group at the Max Planck Institute of Quantum Optics, a.k.a. [Attoworld](https://www.attoworld.de).

The module has several submodules to keep it organized:
- **[data](attoworld/data.html)**: classes and for handling the various data formats used across the labs
- **[numeric](attoworld/numeric.html)**: numerical tools
- **[personal](attoworld/personal.html)**: a module where we can add our own functions that might not be of general interest, but are still good to have available to we can easily share work
- **[plot](attoworld/plot.html)**: functions for plotting with a consistent style
- **[wave](attoworld/wave.html)**: functions for processing waveforms and pulses
- **[spectrum](attoworld/spectrum.html)**: functions for processing spectra
- **[attoworld_rs](attoworld/attoworld_rs.html)**: A place to put Rust code with a Python interface for cases where it's particularly important that the program be fast and correct.
"""

from . import attoworld_rs, data, numeric, personal, plot, spectrum, wave

__all__ = ["data", "numeric", "personal", "plot", "spectrum", "wave", "attoworld_rs"]
