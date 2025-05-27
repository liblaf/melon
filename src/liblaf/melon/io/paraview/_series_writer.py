import contextlib
import os
import shutil
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, overload

import pydantic

from liblaf import grapes
from liblaf.melon.io._save import save


def snake_to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class File(pydantic.BaseModel):
    name: str
    time: float


class Series(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(alias_generator=snake_to_kebab)
    file_series_version: Literal["1.0"] = "1.0"
    files: list[File] = []


class SeriesWriter(Sequence[File], contextlib.AbstractContextManager):
    file: Path
    series: Series
    timestep: float

    def __init__(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        clear: bool = False,
        fps: float = 30.0,
        timestep: float | None = None,
    ) -> None:
        self.file = grapes.as_path(path)
        self.series = Series()
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = 1.0 / fps

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
        return self.file.with_suffix("")

    @property
    def fps(self) -> float:
        return 1.0 / self.timestep

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
        filename: str = f"{self.name}_{len(self):06d}{self.ext}"
        filepath: Path = self.folder / filename
        save(filepath, data)
        if time is None:
            if timestep is None:
                timestep = self.timestep
            time = self.time + timestep
        self.series.files.append(
            File(name=filepath.relative_to(self.file.parent).as_posix(), time=time)
        )
        self.save()

    def end(self) -> None:
        self.save()

    def save(self) -> None:
        grapes.save_pydantic(self.file, self.series, ext=".json", by_alias=True)

    def start(self) -> None:
        pass
