import bisect
import os
from typing import Any

from ._reader import AbstractReader
from ._utils import UnsupportedReaderError


class ReaderDispatcher:
    readers: list[AbstractReader]

    def __init__(self) -> None:
        self.readers = []

    def register(self, reader: AbstractReader) -> None:
        bisect.insort(self.readers, reader, key=lambda r: -r.precedence)

    def load(self, path: str | os.PathLike[str], /, **kwargs) -> Any:
        for reader in self.readers:
            if reader.match_path(path):
                return reader.load(path, **kwargs)
        raise UnsupportedReaderError(path)


reader_dispatcher = ReaderDispatcher()


def register_reader(reader: AbstractReader) -> None:
    reader_dispatcher.register(reader)


def load(path: str | os.PathLike[str], /, **kwargs) -> Any:
    return reader_dispatcher.load(path, **kwargs)
