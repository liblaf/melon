from . import _writer  # noqa: F401
from ._converter import as_multiblock
from ._reader import load_multiblock

__all__ = ["as_multiblock", "load_multiblock"]
