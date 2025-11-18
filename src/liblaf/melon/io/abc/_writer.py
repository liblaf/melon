import collections
import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import attrs
import more_itertools as mit
from liblaf.grapes.logging import depth_logger

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

    def __call__(self, path: PathLike, obj: Any, /, **kwargs) -> None:
        __tracebackhide__ = True
        path = Path(path)
        writer: SingleDispatchCallable[None] | None = self.writers.get(path.suffix)
        if writer is None:
            raise UnsupportedWriterError(type(obj), path)
        path.parent.mkdir(parents=True, exist_ok=True)
        impl: Writer = writer.dispatch(type(obj))
        if impl is _dummy:
            raise UnsupportedWriterError(type(obj), path)
        impl(path, obj, **kwargs)
        depth_logger.debug("Saved %s to '%s'.", type(obj), path)

    def register(
        self, cls: RegType, suffix: str | Iterable[str]
    ) -> Callable[[Writer], Writer]:
        def wrapper(func: Writer) -> Writer:
            for s in mit.always_iterable(suffix):
                self.writers[s].register(cls)(func)
            return func

        return wrapper
