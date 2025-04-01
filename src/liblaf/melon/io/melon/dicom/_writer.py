import os
from pathlib import Path
from typing import Any

from liblaf import melon

from . import as_dicom


class DICOMWriter(melon.io.AbstractWriter):
    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path = Path(path)
        if path.name == "DIRFILE":  # noqa: SIM103
            return True
        return False

    def save(self, path: str | os.PathLike[str], obj: Any) -> None:
        obj: melon.DICOM = as_dicom(obj)
        obj.save(path)
