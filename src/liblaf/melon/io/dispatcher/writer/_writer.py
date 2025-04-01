import abc
import os
from collections.abc import Container
from pathlib import Path
from typing import Any


class AbstractWriter(abc.ABC):
    extensions: Container[str]
    priority: int = 0

    @abc.abstractmethod
    def save(self, path: str | os.PathLike[str], obj: Any) -> None: ...

    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path = Path(path)
        return path.suffix in self.extensions
