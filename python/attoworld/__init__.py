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

## Guidelines
The goal of this module is to gather the python programming work that we do, which maybe others in the group or the community at large could benefit from, into a module that we can easily add to our projects. This is easier if we follow some guidelines for best practices:

 - Use [docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) so that people know how to use your functions.

 - Comment code enough that it's understandable. It's possible that you write in a "self documenting" way, which is fine, but if you're doing something fancy and non-obvious, put in a note

 - If there's a function or class that you think others could benefit from, absolutely feel free to add it to the main modules. If you think you are likely the only one who will use something, you can also add a submodule to the attoworld.personal namespace. This makes it easier to share files with others!

"""

from . import file
from . import numeric
from . import personal
from . import plot
from . import spectrum
from . import wave
from . import attoworld_rs
