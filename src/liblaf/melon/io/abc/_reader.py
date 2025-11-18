from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import attrs
from liblaf.grapes.logging import depth_logger

from liblaf.melon.typing import PathLike


@attrs.define
class UnsupportedReaderError(ValueError):
    path: Path = attrs.field(converter=Path)
    to_type: type

    def __str__(self) -> str:
        return f"Cannot load '{self.path}' as {self.to_type}."


class Reader[T](Protocol):
    def __call__(self, path: Path, /, **kwargs) -> T: ...


@attrs.define
class ReaderDispatcher[T]:
    to_type: type[T]
    registry: dict[str, Reader[T]] = attrs.field(factory=dict)

    def __call__(self, path: PathLike, /, **kwargs) -> T:
        __tracebackhide__ = True
        path = Path(path)
        reader: Reader[T] | None = self.registry.get(path.suffix)
        if reader is None:
            raise UnsupportedReaderError(path, self.to_type)
        obj: T = reader(path, **kwargs)
        depth_logger.debug("Loaded '%s' as %s.", path, type(obj))
        return obj

    def register(self, *suffixes: str) -> Callable[[Reader[T]], Reader[T]]:
        def wrapper(reader: Reader[T]) -> Reader[T]:
            for s in suffixes:
                self.registry[s] = reader
            return reader

        return wrapper
