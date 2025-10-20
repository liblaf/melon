import collections
import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import attrs
from loguru import logger

from liblaf import grapes
from liblaf.melon.typing import PathLike

from ._typing import RegType, SingleDispatchCallable


@attrs.define
class UnsupportedWriterError(ValueError):
    from_type: type
    path: Path = attrs.field(converter=Path)

    def __str__(self) -> str:
        return f"Cannot save {self.from_type} to '{self.path}'."


class Writer(Protocol):
    def __call__(self, path: Path, obj: Any, /, **kwargs) -> None: ...


def _dummy(path: Path, obj: Any, /, **_kwargs) -> None:
    raise UnsupportedWriterError(type(obj), path)


@attrs.define
class WriterDispatcher:
    writers: dict[str, SingleDispatchCallable[None]] = attrs.field(
        factory=lambda: collections.defaultdict(
            lambda: functools.singledispatch(_dummy)
        )
    )

    @grapes.logging.helper
    def __call__(self, path: PathLike, obj: Any, /, **kwargs) -> None:
        path = Path(path)
        writer: SingleDispatchCallable[None] | None = self.writers.get(path.suffix)
        if writer is None:
            raise UnsupportedWriterError(type(obj), path)
        path.parent.mkdir(parents=True, exist_ok=True)
        impl: Writer = writer.dispatch(type(obj))
        if impl is _dummy:
            raise UnsupportedWriterError(type(obj), path)
        impl(path, obj, **kwargs)
        logger.opt(depth=grapes.logging.helper.depth).debug(
            f"Saved {type(obj)} to '{path}'."
        )

    def register(
        self, cls: RegType, suffix: str | Iterable[str]
    ) -> Callable[[Writer], Writer]:
        def wrapper(func: Writer) -> Writer:
            for s in grapes.as_iterable(suffix):
                self.writers[s].register(cls)(func)
            return func

        return wrapper
