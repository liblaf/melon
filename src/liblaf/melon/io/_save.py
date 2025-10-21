from typing import Any

from liblaf.melon.io.abc import WriterDispatcher as _WriterDispatcher
from liblaf.melon.typing import PathLike


class WriterDispatcher(_WriterDispatcher):
    initialized: bool = False

    def __call__(self, path: PathLike, obj: Any, /, **kwargs) -> None:
        __tracebackhide__ = True
        if not self.initialized:
            self.init()
        return super().__call__(path, obj, **kwargs)

    def init(self) -> None:
        # ruff: noqa: F401
        from .pyvista.polydata._write import save_polydata, save_polydata_obj
        from .pyvista.structured_grid._write import save_structured_grid
        from .pyvista.unstructured_grid._write import save_unstructured_grid
        from .trimesh._write import save_trimesh

        self.initialized = True


save = WriterDispatcher()
