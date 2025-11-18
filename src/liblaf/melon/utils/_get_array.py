from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from liblaf.grapes.logging import depth_logger

if TYPE_CHECKING:
    from _typeshed import SupportsContainsAndGetItem


def get_array[KT, VT](
    data: SupportsContainsAndGetItem[KT, VT],
    key: KT,
    deprecated_keys: Iterable[KT] = (),
) -> VT:
    if key in data:
        return data[key]
    for deprecated_name in deprecated_keys:
        if deprecated_name in data:
            depth_logger.warning(
                "'%s' is deprecated. Please use '%s' instead.",
                deprecated_name,
                key,
                stacklevel=2,
            )
            return data[deprecated_name]
    raise KeyError(key)
