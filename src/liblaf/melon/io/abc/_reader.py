from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

import attrs

if TYPE_CHECKING:
    from _typeshed import StrPath


class AbstractReader[T](Protocol):
    """Callable that loads an object from a path."""

    def __call__(self, path: Path, /, **kwargs) -> T: ...


@attrs.define
class ReaderDispatcher[T]:
    """Dispatch readers by file suffix with an optional fallback reader.

    Raises:
        KeyError: If the path suffix is unknown and no fallback reader exists.

    Attributes:
        to_type: Type produced by registered readers.
        fallback: Reader used when no suffix-specific reader is registered.
        registry: Mapping from suffixes such as `.vtp` to reader callables.
    """

    to_type: type[T]
    fallback: AbstractReader[T] | None = None
    registry: dict[str, AbstractReader[T]] = attrs.field(factory=dict)

    def __call__(self, path: StrPath, /, **kwargs) -> T:
        path: Path = Path(path)
        reader: AbstractReader[T] | None = self.registry.get(path.suffix, self.fallback)
        if reader is None:
            raise KeyError(path.suffix)
        return reader(path, **kwargs)

    @overload
    def register(
        self, suffixes: Iterable[str], reader: AbstractReader[T]
    ) -> AbstractReader[T]: ...
    @overload
    def register(
        self, suffixes: Iterable[str], reader: None = None
    ) -> Callable[[AbstractReader[T]], AbstractReader[T]]: ...
    def register(
        self, suffixes: Iterable[str], reader: AbstractReader[T] | None = None
    ) -> Callable[..., Any]:
        if reader is None:
            return functools.partial(self.register, suffixes)
        for s in suffixes:
            self.registry[s] = reader
        return reader

    def register_fallback(self, reader: AbstractReader[T]) -> AbstractReader[T]:
        self.fallback = reader
        return reader
