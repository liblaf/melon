import os
from typing import Any

from . import abc, melon, pyvista

abc.register_writer(melon.DICOMWriter())
abc.register_writer(pyvista.PolyDataWriter())
abc.register_writer(pyvista.UnstructuredGridWriter())


def save(path: str | os.PathLike[str], obj: Any) -> None:
    abc.save(path, obj)
