import abc
import bisect
from collections.abc import Container
from pathlib import Path
from typing import Any

from loguru import logger

from liblaf.melon.typed import PathLike


class UnsupportedWriterError(ValueError):
    data: Any
    path: Path

    def __init__(self, data: Any, path: PathLike, /) -> None:
        self.data = data
        self.path = Path(path)
        super().__init__(f'Cannot save {type(data)} to "{self.path}".')


class AbstractWriter(abc.ABC):
    extensions: Container[str] = ()
    precedence: int = 0

    def __call__(self, path: PathLike, data: Any, /, **kwargs) -> None:
        return self.save(path, data, **kwargs)

    def match_data(self, data: Any, /) -> bool:  # noqa: ARG002
        return True

    def match_path(self, path: PathLike, /) -> bool:
        path = Path(path)
        return path.suffix in self.extensions

    @abc.abstractmethod
    def save(self, path: PathLike, data: Any, /, **kwargs) -> None: ...


class WriterDispatcher:
    writers: list[AbstractWriter]

    def __init__(self) -> None:
        self.writers = []

    def __call__(self, path: PathLike, data: Any, /, **kwargs) -> None:
        return self.save(path, data, **kwargs)

    def register(self, writer: AbstractWriter, /) -> None:
        bisect.insort(self.writers, writer, key=lambda r: -r.precedence)

    def save(self, path: PathLike, data: Any, /, **kwargs) -> None:
        for writer in self.writers:
            if writer.match_data(data) and writer.match_path(path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                writer.save(path, data, **kwargs)
                logger.opt(depth=2).debug('Saved {} to "{}".', type(data), path)
                return
        raise UnsupportedWriterError(data, path)
