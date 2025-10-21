import contextlib
import shutil
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, overload

import attrs
import pydantic

from liblaf import grapes
from liblaf.melon.io._save import save
from liblaf.melon.typing import PathLike


def snake_to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class File(pydantic.BaseModel):
    name: str
    time: float


class Series(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(alias_generator=snake_to_kebab)
    file_series_version: Literal["1.0"] = "1.0"
    files: list[File] = []


@attrs.define
class SeriesWriter(Sequence[File], contextlib.AbstractContextManager):
    file: Path
    series: Series
    step: float

    def __init__(
        self,
        file: PathLike,
        /,
        *,
        clear: bool = False,
        fps: float = 30.0,
        step: float | None = None,
    ) -> None:
        if step is None:
            step = 1.0 / fps
        self.__attrs_init__(file=Path(file), series=Series(), step=step)  # pyright: ignore[reportAttributeAccessIssue]
        if clear:
            shutil.rmtree(self.folder, ignore_errors=True)

    @overload
    def __getitem__(self, index: int) -> File: ...
    @overload
    def __getitem__(self, index: slice) -> list[File]: ...
    def __getitem__(self, index: int | slice) -> File | list[File]:
        return self.series.files[index]

    def __len__(self) -> int:
        return len(self.series.files)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.end()

    @property
    def ext(self) -> str:
        return self.file.suffixes[-2]

    @property
    def folder(self) -> Path:
        return self.file.with_suffix(".d")

    @property
    def fps(self) -> float:
        return 1.0 / self.step

    @property
    def name(self) -> str:
        return self.file.with_suffix("").stem

    @property
    def time(self) -> float:
        if len(self) == 0:
            return 0.0
        return self.series.files[-1].time

    def append(
        self, data: Any, *, time: float | None = None, timestep: float | None = None
    ) -> None:
        __tracebackhide__ = True
        filename: str = f"{self.name}_{len(self):06d}{self.ext}"
        filepath: Path = self.folder / filename
        save(filepath, data)
        if time is None:
            if timestep is None:
                timestep = self.step
            time = self.time + timestep
        self.series.files.append(
            File(name=filepath.relative_to(self.file.parent).as_posix(), time=time)
        )
        self.save()

    def end(self) -> None:
        self.save()

    def save(self) -> None:
        grapes.save(
            self.file, self.series, force_ext=".json", pydantic={"by_alias": True}
        )

    def start(self) -> None:
        pass
