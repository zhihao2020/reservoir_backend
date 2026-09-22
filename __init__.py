"""Thin loader. The inversion payload lives in ``src/`` (source) or ``src.so`` (Linux)."""

from .src import *  # noqa: F403
from .src import __version__
from .src import __all__ as __all__
