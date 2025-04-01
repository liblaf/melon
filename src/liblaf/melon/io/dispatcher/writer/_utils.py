import os
from pathlib import Path


class UnsupportedWriterError(ValueError):
    path: Path

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        super().__init__(f"Cannot save `{self.path}`.")
