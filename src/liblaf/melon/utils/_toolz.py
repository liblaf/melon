import inspect
from collections.abc import Callable, Container, Mapping
from typing import Any

import tlz


def pick[K, V](allowlist: Container[K], dictionary: Mapping[K, V]) -> dict[K, V]:
    return tlz.keyfilter(lambda k: k in allowlist, dictionary)


def filter_kwargs(
    func: Callable[..., Any], kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    from inspect import Parameter

    signature: inspect.Signature = inspect.signature(func)
    filtered: dict[str, Any] = {}
    for p in signature.parameters.values():
        match p.kind:
            case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                if p.name in kwargs:
                    filtered[p.name] = kwargs[p.name]
            case Parameter.VAR_KEYWORD:
                return kwargs
    return filtered
