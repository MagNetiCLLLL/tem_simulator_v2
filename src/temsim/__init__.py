"""TEM Simulator v2 package."""

__version__ = "0.1.0"

# Imported before GUI/NumPy/Numba modules by both application and CLI entries.
from temsim.cpu_resources import initialize_cpu_resources as _initialize_cpu_resources
_initialize_cpu_resources()
