import functools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import overload, override

import attrs
import pyvista as pv

from liblaf import grapes
from liblaf.melon.typing import PathLike

from ._shared import File, Series


@attrs.define
class SeriesReader[T](Sequence[T]):
    file: Path
    loader: Callable[[Path], T]

    def __init__(self, file: PathLike, loader: Callable[[Path], T]) -> None:
        file = Path(file)
        self.__attrs_init__(file=file, loader=loader)  # pyright: ignore[reportAttributeAccessIssue]

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        __tracebackhide__ = True
        files: File | list[File] = self.series.files[index]
        if isinstance(files, File):
            return self.loader(self.folder / files.name)
        return [self.loader(self.folder / f.name) for f in files]

    @override
    def __len__(self) -> int:
        return len(self.series.files)

    @property
    def folder(self) -> Path:
        return self.file.parent

    @functools.cached_property
    def series(self) -> Series:
        return grapes.load(self.file, force_ext=".json", type=Series)

    @property
    def time_values(self) -> list[float]:
        return [f.time for f in self.series.files]

    def _load(self, file: File) -> T:
        obj: T = self.loader(self.folder / file.name)
        if pv.is_pyvista_dataset(obj):
            obj.field_data["Time"] = file.time  # pyright: ignore[reportArgumentType]
        return obj
