import os
from pathlib import Path

from liblaf import melon


def load_dicom(path: str | os.PathLike[str]) -> melon.DICOM:
    return melon.DICOM(path)


class DICOMReader(melon.io.AbstractReader):
    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path = Path(path)
        if path.is_dir() and (path / "DIRFILE").exists():
            return True
        if path.is_file() and path.name == "DIRFILE":  # noqa: SIM103
            return True
        return False

    def load(self, path: str | os.PathLike[str]) -> melon.DICOM:
        return load_dicom(path)
