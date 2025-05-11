from collections.abc import Container
from typing import Any, override

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractWriter
from liblaf.melon.typed import PathLike


class TrimeshWriter(AbstractWriter):
    extensions: Container[str] = {".off", ".ply", ".stl"}

    @override
    def match_data(self, data: Any) -> bool:
        return utils.is_trimesh(data)

    @override
    def save(self, path: PathLike, data: Any, /, **kwargs) -> None:
        import trimesh as tm

        from liblaf.melon.io.trimesh.convert import as_trimesh

        data: tm.Trimesh = as_trimesh(data)
        data.export(path, **kwargs)
