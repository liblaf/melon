from typing import Any


def abbr_type_name(cls: Any) -> str:
    if not isinstance(cls, type):
        cls = type(cls)
    module_name: str = cls.__module__.split(".", maxsplit=1)[0]
    return f"{module_name}.{cls.__name__}"
