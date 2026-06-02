from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, Protocol, overload

import attrs

if TYPE_CHECKING:
    from functools import _RegType, _SingleDispatchCallable


class AbstractConverter[F, T](Protocol):
    def __call__(self, obj: F, /, **kwargs) -> T: ...


def _default_converter(obj: Any, /, **kwargs) -> Never:
    raise NotImplementedError


def _identity[T](obj: T, /, **kwargs) -> T:
    del kwargs
    return obj


@attrs.define
class ConverterDispatcher[T]:
    def _default_registry(self) -> _SingleDispatchCallable[T]:
        registry: _SingleDispatchCallable[T] = functools.singledispatch(
            _default_converter
        )
        registry.register(self.to_type, _identity)
        return registry

    to_type: type[T]
    registry: _SingleDispatchCallable[T] = attrs.field(
        default=attrs.Factory(_default_registry, takes_self=True)
    )

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
