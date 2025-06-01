"""
Tools from the Attosecond science group at the Max Planck Institute of Quantum Optics, a.k.a. [Attoworld](https://www.attoworld.de)

## Structure
The module has several submodules to keep it organized:
- *file*: functions for reading/writing the various data formats used across the labs
- *numeric*: numerical tools
- *personal*: a module where we can add our own functions that might not be of general interest, but are still good to have available to we can easily share work
- *plot*: functions for plotting with a consistent style
- *wave*: functions for processing waveforms
- *spectrum*: functions for processing spectra
- *attoworld_rs*: A place to put Rust code with a Python interface for cases where it's particularly important that the program be fast and correct.
"""

from . import file
from . import numeric
from . import personal
from . import plot
from . import spectrum
from . import wave
from . import attoworld_rs
