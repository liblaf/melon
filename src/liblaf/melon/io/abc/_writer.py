from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import attrs

from liblaf.melon.typing import PathLike


@attrs.define
class UnsupportedWriterError(ValueError):
    from_type: type
    path: Path = attrs.field(converter=Path)

    def __str__(self) -> str:
        return f"Cannot save {self.from_type} to '{self.path}'."


class Writer(Protocol):
    @property
    def suffixes(self) -> Iterable[str]: ...
    def __call__(self, path: Path, obj: Any, /, **kwargs) -> None: ...


@attrs.define
class WriterDispatcher:
    writers: dict[str, Writer] = attrs.field(factory=dict)

    def __call__(self, path: PathLike, data: Any, /, **kwargs) -> None:
        path = Path(path)
        writer: Writer | None = self.writers.get(path.suffix)
        if writer is None:
            raise UnsupportedWriterError(type(data), path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path, data, **kwargs)

    def register(self, writer: Writer) -> None:
        for s in writer.suffixes:
            self.writers[s] = writer
