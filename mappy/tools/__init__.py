"""
=========================
Useful tools for analysis
=========================

.. autosummary::
    :toctree: generated/

    continuation

"""

from . import _continuation
from ._continuation import *

__all__ = []
__all__.extend(_continuation.__all__.copy())