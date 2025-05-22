import abc
import bisect
from collections.abc import Container
from pathlib import Path

from loguru import logger

from liblaf.melon.typed import PathLike


class UnsupportedReaderError(ValueError):
    dtype: type
    path: Path

    def __init__(self, dtype: type, path: PathLike, /) -> None:
        self.dtype = dtype
        self.path = Path(path)
        super().__init__(f'Cannot load "{self.path}" as {dtype}.')


class AbstractReader[T](abc.ABC):
    extensions: Container[str] = ()
    precedence: int = 0

    def __call__(self, path: PathLike, /, **kwargs) -> T:
        return self.load(path, **kwargs)

    @abc.abstractmethod
    def load(self, path: PathLike, /, **kwargs) -> T: ...

    def match_path(self, path: PathLike, /) -> bool:
        path = Path(path)
        return path.suffix in self.extensions


class ReaderDispatcher[T]:
    readers: list[AbstractReader]
    dtype: type[T]

    def __init__(self, dtype: type[T], /) -> None:
        self.dtype = dtype
        self.readers = []

    def __call__(self, path: PathLike, /, **kwargs) -> T:
        return self.load(path, **kwargs)

    def load(self, path: PathLike, /, **kwargs) -> T:
        path = Path(path)
        for reader in self.readers:
            if reader.match_path(path):
                data: T = reader.load(path, **kwargs)
                logger.opt(depth=2).debug('Loaded {} from "{}".', type(data), path)
                return data
        raise UnsupportedReaderError(self.dtype, path)

    def register(self, reader: AbstractReader, /) -> None:
        bisect.insort(self.readers, reader, key=lambda r: -r.precedence)
