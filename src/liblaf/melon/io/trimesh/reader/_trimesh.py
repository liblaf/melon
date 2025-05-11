from pathlib import Path
from typing import Any, override

from liblaf.melon.io.abc import AbstractReader
from liblaf.melon.typed import PathLike


class TrimeshReader(AbstractReader):
    @override
    def load(self, path: PathLike, /, **kwargs) -> Any:
        import trimesh as tm

        return tm.load(Path(path), **kwargs)
