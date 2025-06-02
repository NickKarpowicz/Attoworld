"""
This module will contain data processing routines that operate on measured or simulated waveforms.
"""

from .wave import align_waves, get_effective_response
from .trace_handler import TraceHandler, MultiTraceHandler
__all__ = [
    "align_waves",
    "get_effective_response",
    "TraceHandler",
    "MultiTraceHandler"
]
