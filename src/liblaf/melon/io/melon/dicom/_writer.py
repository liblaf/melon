import os
from pathlib import Path
from typing import Any, override

from liblaf import grapes
from liblaf.melon import struct
from liblaf.melon.io import abc

from .conversion import as_dicom


class DICOMWriter(abc.AbstractWriter):
    @override
    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path: Path = grapes.as_path(path)
        if path.name == "DIRFILE":  # noqa: SIM103
            return True
        return False

    @override
    def save(self, path: str | os.PathLike[str], obj: Any, /, **kwargs) -> None:
        obj: struct.DICOM = as_dicom(obj)
        obj.save(path)
