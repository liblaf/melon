import functools
from collections.abc import Callable
from typing import Any

import attrs

from ._typing import RegType, SingleDispatchCallable


@attrs.define
class UnsupportedConverterError(ValueError):
    from_type: type
    to_type: type

    def __init__(self, obj_or_cls: Any, to_type: type, /) -> None:
        from_type: type = (
            obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
        )
        self.__attrs_init__(from_type=from_type, to_type=to_type)  # pyright: ignore[reportAttributeAccessIssue]

    def __str__(self) -> str:
        return f"Cannot convert {self.from_type} to {self.to_type}."


@attrs.define
class ConverterDispatcher[T]:
    to_type: type[T]
    dispatch: SingleDispatchCallable[T]

    def __init__(self, to_type: type[T]) -> None:
        @functools.singledispatch
        def dispatch(obj: Any, /, **kwargs) -> T:  # noqa: ARG001
            raise UnsupportedConverterError(obj, to_type)

        self.__attrs_init__(to_type=to_type, dispatch=dispatch)  # pyright: ignore[reportAttributeAccessIssue]

    def __call__(self, obj: Any, /, **kwargs) -> T:
        if isinstance(obj, self.to_type):
            return obj
        result: T = self.dispatch(obj, **kwargs)
        # logger.trace(f"Converted {type(obj)} to {type(result)}.")
        return result

    def register(self, cls: RegType) -> Callable[[Callable[..., T]], Callable[..., T]]:
        return self.dispatch.register(cls)
