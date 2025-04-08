import abc
import os
from collections.abc import Container
from pathlib import Path
from typing import Any


class AbstractReader(abc.ABC):
    extensions: Container[str]
    precedence: int = 0

    @abc.abstractmethod
    def load(self, path: str | os.PathLike[str], /, **kwargs) -> Any: ...

    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path = Path(path)
        return path.suffix in self.extensions
