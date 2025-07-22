import abc
import bisect
from typing import Any

from liblaf.grapes.typed import ClassInfo


class UnsupportedConverterError(ValueError):
    data: Any
    to_type: type

    def __init__(self, data: Any, to_type: type, /) -> None:
        self.data = data
        self.to_type = to_type
        super().__init__(f"Cannot convert {type(data)} to {to_type}.")


class AbstractConverter[T](abc.ABC):
    from_type: ClassInfo = ()
    precedence: int = 0

    def __call__(self, data: Any, /, **kwargs) -> T:
        return self.convert(data, **kwargs)

    @abc.abstractmethod
    def convert(self, data: Any, /, **kwargs) -> T: ...

    def match_from(self, data: Any, /) -> bool:
        return isinstance(data, self.from_type)


class ConverterDispatcher[T]:
    converters: list[AbstractConverter]
    to_type: type[T]

    def __init__(self, to_type: type[T], /) -> None:
        self.converters = []
        self.to_type = to_type

    def __call__(self, data: Any, /, **kwargs) -> T:
        return self.convert(data, **kwargs)

    def convert(self, data: Any, /, **kwargs) -> T:
        if isinstance(data, self.to_type):
            return data
        for converter in self.converters:
            if converter.match_from(data):
                return converter.convert(data, **kwargs)
        raise UnsupportedConverterError(data, self.to_type)

    def register(self, converter: AbstractConverter, /) -> None:
        bisect.insort(self.converters, converter, key=lambda c: -c.precedence)
