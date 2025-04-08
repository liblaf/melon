import bisect
from typing import Any

from ._converter import AbstractConverter
from ._utils import UnsupportedConversionError


class ConversionDispatcher:
    converters: list[AbstractConverter]

    def __init__(self) -> None:
        self.converters = []

    def register(self, converter: AbstractConverter) -> None:
        bisect.insort(self.converters, converter, key=lambda c: -c.precedence)

    def convert[T](self, obj: Any, type_to: type[T], /, **kwargs) -> T:
        if isinstance(obj, type_to):
            return obj
        for converter in self.converters:
            if converter.match_from(obj) and converter.match_to(type_to):
                return converter.convert(obj, **kwargs)
        raise UnsupportedConversionError(obj, type_to)


conversion_dispatcher = ConversionDispatcher()


def register_converter(converter: AbstractConverter) -> None:
    conversion_dispatcher.register(converter)


def convert[T](obj: Any, type_to: type[T], /, **kwargs) -> T:
    return conversion_dispatcher.convert(obj, type_to, **kwargs)
