import bisect
import os
from typing import Any

from ._utils import UnsupportedWriterError
from ._writer import AbstractWriter


class WriterDispatcher:
    writers: list[AbstractWriter]

    def __init__(self) -> None:
        self.writers = []

    def register(self, writer: AbstractWriter) -> None:
        bisect.insort(self.writers, writer, key=lambda r: -r.precedence)

    def save(self, path: str | os.PathLike[str], obj: Any, /, **kwargs) -> None:
        for writer in self.writers:
            if writer.match_path(path):
                return writer.save(path, obj, **kwargs)
        raise UnsupportedWriterError(path)


writer_dispatcher = WriterDispatcher()


def register_writer(writer: AbstractWriter) -> None:
    writer_dispatcher.register(writer)


def save(path: str | os.PathLike[str], obj: Any, /, **kwargs) -> None:
    return writer_dispatcher.save(path, obj, **kwargs)
