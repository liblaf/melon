import contextlib
import os
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
    path: Path
    series: Series
    timestep: float

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fps: float = 30.0,
        timestep: float | None = None,
    ) -> None:
        self.path = grapes.as_path(path)
        self.series = Series()
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = 1.0 / fps

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
        return self.path.suffixes[-2]

    @property
    def fps(self) -> float:
        return 1.0 / self.timestep

    @property
    def frames_dir(self) -> Path:
        return self.path.parent / self.name

    @property
    def name(self) -> str:
        return self.path.with_suffix("").stem

    @property
    def time(self) -> float:
        if len(self) == 0:
            return 0.0
        return self.series.files[-1].time

    def append(
        self,
        data: Any,
        *,
        time: float | None = None,
        timestep: float | None = None,
    ) -> None:
        filename: str = f"F{len(self):06d}{self.ext}"
        filepath: Path = self.frames_dir / filename
        save(filepath, data)
        self.save()
        if time is None:
            if timestep is None:
                timestep = self.timestep
            time = self.time + timestep
        self.series.files.append(
            File(name=filepath.relative_to(self.path.parent).as_posix(), time=time)
        )

    def end(self) -> None:
        self.save()

    def save(self) -> None:
        grapes.save_json(self.path, self.series.model_dump(by_alias=True))

    def start(self) -> None:
        pass
