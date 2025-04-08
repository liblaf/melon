import os
from pathlib import Path
from typing import override

from liblaf import grapes
from liblaf.melon import struct
from liblaf.melon.io import abc


def load_dicom(path: str | os.PathLike[str], /, **kwargs) -> struct.DICOM:
    return struct.DICOM(path, **kwargs)


class DICOMReader(abc.AbstractReader):
    @override
    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path: Path = grapes.as_path(path)
        if path.is_dir() and (path / "DIRFILE").exists():
            return True
        if path.is_file() and path.name == "DIRFILE":  # noqa: SIM103
            return True
        return False

    @override
    def load(self, path: str | os.PathLike[str], /, **kwargs) -> struct.DICOM:
        return load_dicom(path, **kwargs)
