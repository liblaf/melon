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
        bisect.insort(self.readers, reader, key=lambda r: r.priority)

    def load(self, path: str | os.PathLike[str]) -> Any:
        for reader in self.readers:
            if reader.match_path(path):
                return reader.load(path)
        raise UnsupportedReaderError(path)


reader_dispatcher = ReaderDispatcher()
register_reader = reader_dispatcher.register
load = reader_dispatcher.load
