from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

import attrs

if TYPE_CHECKING:
    from functools import _RegType, _SingleDispatchCallable

    from _typeshed import StrPath


class AbstractWriter[T](Protocol):
    def __call__(self, obj: T, path: Path, /, **kwargs) -> None: ...


def _default_writer(obj: Any, path: Path, /, **kwargs) -> None:
    raise NotImplementedError


@attrs.define
class WriterDispatcher[T]:
    registry: dict[str, _SingleDispatchCallable[None]] = attrs.field(factory=dict)

    def __call__(self, obj: T, path: StrPath, /, **kwargs) -> None:
        path: Path = Path(path)
        writer: _SingleDispatchCallable[None] = self.registry[path.suffix]
        path.parent.mkdir(parents=True, exist_ok=True)
        return writer(obj, path, **kwargs)

    @overload
    def register(
        self, cls: _RegType, suffixes: Iterable[str], writer: AbstractWriter[T]
    ) -> AbstractWriter[T]: ...
    @overload
    def register(
        self, cls: _RegType, suffixes: Iterable[str], writer: None = None
    ) -> Callable[[AbstractWriter[T]], AbstractWriter[T]]: ...
    def register(
        self,
        cls: _RegType,
        suffixes: Iterable[str],
        writer: AbstractWriter[T] | None = None,
    ) -> Callable[..., Any]:
        if writer is None:
            return functools.partial(self.register, cls, suffixes)
        for suffix in suffixes:
            if suffix not in self.registry:
                self.registry[suffix] = functools.singledispatch(_default_writer)
            self.registry[suffix].register(cls, writer)
        return writer


save: WriterDispatcher[Any] = WriterDispatcher()
