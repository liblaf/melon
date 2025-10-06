import types
from collections.abc import Callable
from typing import Any, Protocol, overload

type RegType = type[Any] | types.UnionType


class SingleDispatchCallable[T](Protocol):
    registry: types.MappingProxyType[Any, Callable[..., T]]

    def dispatch(self, cls: Any) -> Callable[..., T]: ...
    # @fun.register(complex)
    # def _(arg, verbose=False): ...
    @overload
    def register(
        self, cls: RegType, func: None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
    # @fun.register
    # def _(arg: int, verbose=False):
    @overload
    def register(
        self, cls: Callable[..., T], func: None = None
    ) -> Callable[..., T]: ...
    # fun.register(int, lambda x: x)
    @overload
    def register(self, cls: RegType, func: Callable[..., T]) -> Callable[..., T]: ...
    def _clear_cache(self) -> None: ...
    def __call__(self, /, *args: Any, **kwargs: Any) -> T: ...
