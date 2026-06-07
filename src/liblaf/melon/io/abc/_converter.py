from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, Protocol, overload

import attrs

if TYPE_CHECKING:
    from functools import _RegType, _SingleDispatchCallable


class AbstractConverter[F, T](Protocol):
    """Callable that converts one object into another mesh representation."""

    def __call__(self, obj: F, /, **kwargs) -> T: ...


def _default_converter(obj: Any, /, **kwargs) -> Never:
    raise NotImplementedError


def _identity[T](obj: T, /, **kwargs) -> T:
    del kwargs
    return obj


@attrs.define
class ConverterDispatcher[T]:
    """Single-dispatch conversion registry with identity conversion built in.

    The target type is registered as an identity conversion, so callers can pass
    through objects that are already in the requested representation.

    Examples:
        >>> converter = ConverterDispatcher(str)
        >>> converter("mesh")
        'mesh'
        >>> @converter.register(int)
        ... def _from_int(obj, /, **kwargs):
        ...     return str(obj)
        >>> converter(42)
        '42'
    """

    def _default_registry(self) -> _SingleDispatchCallable[T]:
        registry: _SingleDispatchCallable[T] = functools.singledispatch(
            _default_converter
        )
        registry.register(self.to_type, _identity)
        return registry

    to_type: type[T]
    """Target type returned unchanged when the input already has that runtime type."""
    registry: _SingleDispatchCallable[T] = attrs.field(
        default=attrs.Factory(_default_registry, takes_self=True)
    )
    """Underlying `functools.singledispatch` registry."""

    def __call__(self, obj: Any, /, **kwargs) -> T:
        return self.registry(obj, **kwargs)

    @overload
    def register[F](
        self, cls: _RegType, converter: AbstractConverter[F, T]
    ) -> AbstractConverter[F, T]: ...
    @overload
    def register[F](
        self, cls: _RegType, converter: None = None
    ) -> Callable[[AbstractConverter[F, T]], AbstractConverter[F, T]]: ...
    def register[F](
        self, cls: _RegType, converter: AbstractConverter[F, T] | None = None
    ) -> Callable[..., Any]:
        if converter is None:
            return functools.partial(self.register, cls)
        self.registry.register(cls, converter)
        return converter
