from pathlib import Path
from typing import Any, override

from liblaf.melon.io.abc import Reader
from liblaf.melon.typing import PathLike


class TrimeshReader(Reader):
    @override
    def load(self, path: PathLike, /, **kwargs) -> Any:
        import trimesh as tm

        return tm.load(Path(path), **kwargs)
