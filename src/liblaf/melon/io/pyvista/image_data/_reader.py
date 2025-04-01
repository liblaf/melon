from pathlib import Path

import pyvista as pv

import liblaf.grapes as grapes  # noqa: PLR0402
from liblaf import melon
from liblaf.melon.typed import PathLike


def load_image_data(path: PathLike) -> pv.ImageData:
    path = Path(path)
    if path.is_file() and path.name == "DIRFILE":
        path = path.parent
    if path.is_dir() and (path / "DIRFILE").exists():
        return pv.read(path, force_ext=".dcm")  # pyright: ignore[reportReturnType]
    return pv.read(path)  # pyright: ignore[reportReturnType]


class ImageDataReader(melon.io.AbstractReader):
    def match_path(self, path: PathLike) -> bool:
        path: Path = grapes.as_path(path)
        if path.is_file() and path.name == "DIRFILE":
            return True
        if path.is_dir() and (path / "DIRFILE").exists():
            return True
        return path.suffix in {".dcm", ".vti"}

    def load(self, path: PathLike) -> pv.ImageData:
        return load_image_data(path)
